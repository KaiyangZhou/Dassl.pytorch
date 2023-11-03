import torch
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import Conv2dDynamic

__all__ = ["RobustDDG"]

PE_CI = "Cross-Instance"
PE_CK = "Cross-Kernel"
PE_TYPES = [PE_CI, PE_CK]
TARGET_PE = None
PE_ON = False


def shuffle_column(data):
    cdata = data.clone()
    B = data.shape[0]
    C = data.shape[1]
    for i in range(B):
        ridxs = torch.randperm(C)
        cdata[i] = data[i][ridxs]
    return cdata


def pe_forward(self, x, attention_x=None):
    attention_x = x if attention_x is None else attention_x
    y = self.attention(attention_x)

    if PE_ON:
        if TARGET_PE == PE_CI:
            # CI-PE
            rand_idxs = torch.randperm(y.size(0), device=y.device)
            y = y[rand_idxs]
        elif TARGET_PE == PE_CK:
            # CK-PE
            y = shuffle_column(y)
        else:
            raise ValueError(f"Available PEs are:{PE_TYPES}")

    out = self.conv(x)

    for i, template in enumerate(self.kernel_templates):
        out += self.kernel_templates[template](x) * y[:, i].view(-1, 1, 1, 1)

    return out


@TRAINER_REGISTRY.register()
class RobustDDG(TrainerX):
    """RobustDDG.
    
    Parameter Exchange for Robust Dynamic Domain Generalization.

    https://github.com/MetaVisionLab/PE
    """

    def __init__(self, cfg):
        super(RobustDDG, self).__init__(cfg)
        self.available_backbones = \
            ["resnet18_dynamic", "resnet50_dynamic", "resnet101_dynamic"]
        assert cfg.MODEL.BACKBONE.NAME in self.available_backbones, \
            f"PE method supports these backbones: {self.available_backbones}"
        self.swa_model = AveragedModel(self.model)
        self.register_model("swa_model", self.swa_model, None, None)
        # you can change the PE type by setting the TRAINER.ROBUSTDDG.TYPE
        assert cfg.TRAINER.ROBUSTDDG.TYPE in PE_TYPES, \
            f"Available PEs are:{PE_TYPES}"
        global TARGET_PE, PE_ON
        TARGET_PE = cfg.TRAINER.ROBUSTDDG.TYPE
        PE_ON = False
        # inject PE
        Conv2dDynamic.forward = pe_forward

    def model_inference(self, input):
        global PE_ON
        PE_ON = False  # always False for inference
        # use the SWA model for inference
        return self.swa_model(input)

    def forward_backward(self, batch):
        images, labels, _ = self.parse_batch_train(batch)
        raw_output = self.model(images)
        global PE_ON
        PE_ON = True
        perturbed_output = self.model(images)
        PE_ON = False
        loss = F.cross_entropy(raw_output, labels) \
            + F.cross_entropy(perturbed_output, labels)
        self.model_backward_and_update(loss)

        # update BN statistics for the SWA model
        with torch.no_grad():
            self.swa_model(images)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(raw_output, labels)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.swa_model.update_parameters(self.model)
            self.update_lr()

        return loss_summary
