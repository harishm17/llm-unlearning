import copy
from trainer.utils import compute_kl_divergence
from trainer.unlearn.base import UnlearnTrainer


class GradDiff(UnlearnTrainer):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.ref_model = None
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        # Keep ref_model on CPU to avoid doubling GPU memory (8B params = ~16 GB).
        # compute_dpo_loss moves inputs to ref_device automatically.
        ref_model = copy.deepcopy(model).to("cpu")
        ref_model.eval()
        return ref_model

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        if self.retain_loss_type == "NLL":
            retain_loss += retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += kl_loss
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )
        return retain_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss
