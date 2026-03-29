"""
MultiTurnEvaluator: runs the MT-Eval harness on a trained model.

Usage:
    python src/eval/mt_eval.py \
        --checkpoint saves/unlearn/MTSimNPO_forget10/checkpoint-final \
        --mt_test_path data/mt_test.jsonl \
        --split test \
        --output results/M1_seed0/mt_test.json \
        [--examples_output results/M1_seed0/mt_test_examples.jsonl]  # per-example log
        [--llm_judge]  # costs ~$0.25 per full eval run
"""
from __future__ import annotations
import json, argparse, torch
from pathlib import Path
from statistics import mean
from typing import Optional
from sentence_transformers import SentenceTransformer
import spacy

try:
    # When run as `PYTHONPATH=src python src/eval/mt_eval.py`
    from eval.mt_metrics import (
        normalized_entity_match, semantic_similarity,
        is_leaked, make_llm_judge,
    )
except ModuleNotFoundError:
    from src.eval.mt_metrics import (
        normalized_entity_match, semantic_similarity,
        is_leaked, make_llm_judge,
    )

ATTACK_TYPES_TRAIN    = ["priming", "self_correction", "persona_switch",
                         "priming_v2", "self_correction_v2", "persona_switch_v2"]
ATTACK_TYPES_TRANSFER = ["cot_decomposition", "triangulation"]
ATTACK_TYPES_STRESS   = ["crescendo"]
ALL_ATTACK_TYPES      = ATTACK_TYPES_TRAIN + ATTACK_TYPES_TRANSFER + ATTACK_TYPES_STRESS


class MultiTurnEvaluator:
    def __init__(self, model, tokenizer, mt_test_path: str,
                 llm_judge: bool = True, openai_client=None):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = SentenceTransformer("all-mpnet-base-v2")
        self.nlp = spacy.load("en_core_web_sm")
        self.llm_judge_fn = (
            make_llm_judge(openai_client) if llm_judge and openai_client else None
        )
        with open(mt_test_path) as f:
            self.test_data = [json.loads(line) for line in f if line.strip()]

    def _generate(self, conversation_prefix: list, max_new_tokens: int = 150) -> str:
        """Run model generation given a conversation prefix (list of role/content dicts)."""
        prompt = self.tokenizer.apply_chat_template(
            conversation_prefix, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )

    def evaluate(self, split: str = "test",
                 examples_output: Optional[str] = None) -> dict:
        """
        Evaluate all examples with the given split label.
        Returns summary dict with MTRR, KLT, and sub-scores per attack type.

        Args:
            split: "val" or "test"
            examples_output: if set, writes per-example JSONL to this path
        """
        data = [d for d in self.test_data if d.get("split") == split]
        results = {a: {"leaked": [], "rouge": [], "nem": [], "sem": [],
                        "judge": [], "klt": []} for a in ALL_ATTACK_TYPES}
        per_example_records = [] if examples_output else None

        from rouge_score import rouge_scorer as rs
        rouge = rs.RougeScorer(["rougeL"], use_stemmer=True)

        for example in data:
            attack = example["attack_type"]
            answer = example["answer"]
            topic  = example.get("author_name", answer.split()[0])
            conv   = example["conversation"]

            if attack not in results:
                continue

            # Evaluate at each intermediate assistant turn (to compute KLT)
            leaked_at_turn = None
            for turn_idx in range(2, len(conv), 2):
                prefix = conv[:turn_idx]
                gen = self._generate(prefix)
                if (leaked_at_turn is None and
                        is_leaked(gen, answer, topic, self.nlp,
                                  self.encoder, self.llm_judge_fn)):
                    leaked_at_turn = turn_idx // 2

            # Full conversation eval (primary metric)
            full_prefix = conv[:-1]  # all but final assistant turn
            full_gen = self._generate(full_prefix)

            rouge_l = rouge.score(answer, full_gen)["rougeL"].fmeasure
            nem  = normalized_entity_match(full_gen, answer, self.nlp)
            sem  = semantic_similarity(full_gen, answer, self.encoder)
            judge = self.llm_judge_fn(full_gen, topic) if self.llm_judge_fn else 0.0
            leaked = is_leaked(full_gen, answer, topic, self.nlp,
                               self.encoder, self.llm_judge_fn)

            results[attack]["leaked"].append(leaked)
            results[attack]["rouge"].append(rouge_l)
            results[attack]["nem"].append(nem)
            results[attack]["sem"].append(sem)
            results[attack]["judge"].append(judge)
            if leaked_at_turn is not None:
                results[attack]["klt"].append(leaked_at_turn)

            if per_example_records is not None:
                per_example_records.append({
                    "author_name":  topic,
                    "attack_type":  attack,
                    "leaked":       leaked,
                    "nem":          round(nem, 4),
                    "sem":          round(sem, 4),
                    "rouge":        round(rouge_l, 4),
                    "judge":        judge,
                    "klt":          leaked_at_turn,
                    "generated":    full_gen[:300],   # truncate for storage
                })

        # Aggregate
        summary = {}
        for attack in ALL_ATTACK_TYPES:
            r = results[attack]
            if not r["leaked"]:
                continue
            n = len(r["leaked"])
            summary[attack] = {
                "mtrr":      sum(r["leaked"]) / n,
                "avg_rouge": mean(r["rouge"]),
                "avg_nem":   mean(r["nem"]),
                "avg_sem":   mean(r["sem"]),
                "avg_judge": mean(r["judge"]),
                "avg_klt":   mean(r["klt"]) if r["klt"] else None,
                "n": n,
            }

        def _mean_mtrr(attack_list):
            vals = [summary[a]["mtrr"] for a in attack_list if a in summary]
            return mean(vals) if vals else None

        summary["overall_mtrr_trained"]  = _mean_mtrr(ATTACK_TYPES_TRAIN)
        summary["overall_mtrr_transfer"] = _mean_mtrr(ATTACK_TYPES_TRANSFER)
        summary["overall_mtrr_stress"]   = _mean_mtrr(ATTACK_TYPES_STRESS)

        if per_example_records is not None:
            Path(examples_output).parent.mkdir(parents=True, exist_ok=True)
            with open(examples_output, "w") as f:
                for rec in per_example_records:
                    f.write(json.dumps(rec) + "\n")

        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mt_test_path", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--llm_judge", action="store_true")
    parser.add_argument("--openai_key", default=None)
    parser.add_argument("--examples_output", default=None,
                        help="Optional path to write per-example JSONL results")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import openai

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    openai_client = openai.OpenAI(api_key=args.openai_key) if args.openai_key else None

    evaluator = MultiTurnEvaluator(
        model=model, tokenizer=tokenizer,
        mt_test_path=args.mt_test_path,
        llm_judge=args.llm_judge,
        openai_client=openai_client,
    )
    results = evaluator.evaluate(split=args.split, examples_output=args.examples_output)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    def _fmt(v):
        return f"{v:.3f}" if v is not None else "N/A"

    print(f"Overall MTRR (trained attacks):  {_fmt(results.get('overall_mtrr_trained'))}")
    print(f"Overall MTRR (transfer attacks): {_fmt(results.get('overall_mtrr_transfer'))}")
    print(f"Overall MTRR (stress attacks):   {_fmt(results.get('overall_mtrr_stress'))}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
