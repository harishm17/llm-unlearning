from transformers import AutoModelForCausalLM
from trainer.utils import compute_dpo_loss
from trainer.unlearn.grad_diff import GradDiff


class NPO(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        # Load ref_model in 4-bit quantization on GPU (~4 GB vs ~16 GB for bf16).
        # This keeps inference fast while freeing ~12 GB VRAM vs a full bf16 copy.
        from transformers import BitsAndBytesConfig
        pretrained_path = model.config._name_or_path
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model.dtype,
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        ref_model.eval()
        return ref_model

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]

        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
