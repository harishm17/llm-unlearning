"""
MTSimNPO: SimNPO extended with multi-turn forget examples.

Loss:
    L = gamma * L_SimNPO(D_f)                    [single-turn forget]
      + gamma * mt_weight * L_SimNPO(D_mt)       [multi-turn forget]
      + alpha * L_NLL(D_r)                        [retain regularization]

The SimNPO loss per batch:
    nll_norm = nll_per_seq / response_token_count - delta
    loss = -(2/beta) * logsigmoid(beta * nll_norm).mean()

For multi-turn: labels mask ALL prefix tokens with -100.
|y| = loss_mask.sum(-1) = only final assistant response token count.
This is enforced by MultiTurnForgetCollator — the trainer formula is identical.
"""
import torch
import torch.nn.functional as F
from trainer.unlearn.simnpo import SimNPO
from trainer.utils import compute_batch_nll


class MTSimNPO(SimNPO):
    """SimNPO with an added multi-turn forget loss term."""

    def __init__(self, *args, mt_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.mt_weight = mt_weight

    def _simnpo_loss(self, nll_per_seq: torch.Tensor,
                     loss_mask_sum: torch.Tensor) -> torch.Tensor:
        """Compute SimNPO loss from per-sequence NLL and response token counts."""
        nll_norm = nll_per_seq / loss_mask_sum - self.delta
        return -F.logsigmoid(self.beta * nll_norm).mean() * 2 / self.beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # ── Single-turn forget ─────────────────────────────────────────────
        forget_inputs = inputs["forget"]
        loss_mask_st = (forget_inputs["labels"] != -100)
        forget_nll, forget_outputs = compute_batch_nll(model, forget_inputs)
        st_loss = self._simnpo_loss(forget_nll, loss_mask_st.sum(-1).float())

        # ── Multi-turn forget ──────────────────────────────────────────────
        mt_loss = torch.zeros(1, device=st_loss.device, dtype=st_loss.dtype)
        if inputs.get("mt_forget") is not None:
            mt_inputs = inputs["mt_forget"]
            loss_mask_mt = (mt_inputs["labels"] != -100)
            mt_nll, _ = compute_batch_nll(model, mt_inputs)
            mt_loss = self._simnpo_loss(mt_nll, loss_mask_mt.sum(-1).float())

        # ── Retain regularization ──────────────────────────────────────────
        retain_inputs = {k: inputs["retain"][k]
                         for k in ("input_ids", "attention_mask", "labels")}
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = (self.gamma * st_loss
                + self.gamma * self.mt_weight * mt_loss
                + self.alpha * retain_loss)

        self.log({
            "st_loss": st_loss.item(),
            "mt_loss": mt_loss.item(),
            "retain_loss": retain_loss.item(),
        })

        return (loss, forget_outputs) if return_outputs else loss
