import torch
from torch.nn import functional as F

from dassl.data import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.data.transforms import build_transform


@TRAINER_REGISTRY.register()
class FixMatch(TrainerXU):
    """FixMatch: Simplifying Semi-Supervised Learning with
    Consistency and Confidence.

    https://arxiv.org/abs/2001.07685.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.weight_u = cfg.TRAINER.FIXMATCH.WEIGHT_U
        self.conf_thre = cfg.TRAINER.FIXMATCH.CONF_THRE

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.FIXMATCH.STRONG_TRANSFORMS) > 0

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.FIXMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        self.dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = self.dm.train_loader_x
        self.train_loader_u = self.dm.train_loader_u
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel() # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {
            'acc_thre': acc_thre,
            'acc_raw': acc_raw,
            'keep_rate': keep_rate
        }
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_data = self.parse_batch_train(batch_x, batch_u)
        input_x, input_x2, label_x, input_u, input_u2, label_u = parsed_data
        input_u = torch.cat([input_x, input_u], 0)
        input_u2 = torch.cat([input_x2, input_u2], 0)
        n_x = input_x.size(0)

        # Generate pseudo labels
        with torch.no_grad():
            output_u = F.softmax(self.model(input_u), 1)
            max_prob, label_u_pred = output_u.max(1)
            mask_u = (max_prob >= self.conf_thre).float()

            # Evaluate pseudo labels' accuracy
            y_u_pred_stats = self.assess_y_pred_quality(
                label_u_pred[n_x:], label_u, mask_u[n_x:]
            )

        # Supervised loss
        output_x = self.model(input_x)
        loss_x = F.cross_entropy(output_x, label_x)

        # Unsupervised loss
        output_u = self.model(input_u2)
        loss_u = F.cross_entropy(output_u, label_u_pred, reduction='none')
        loss_u = (loss_u * mask_u).mean()

        loss = loss_x + loss_u * self.weight_u
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss_x': loss_x.item(),
            'acc_x': compute_accuracy(output_x, label_x)[0].item(),
            'loss_u': loss_u.item(),
            'y_u_pred_acc_raw': y_u_pred_stats['acc_raw'],
            'y_u_pred_acc_thre': y_u_pred_stats['acc_thre'],
            'y_u_pred_keep': y_u_pred_stats['keep_rate']
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x['img']
        input_x2 = batch_x['img2']
        label_x = batch_x['label']
        input_u = batch_u['img']
        input_u2 = batch_u['img2']
        # label_u is used only for evaluating pseudo labels' accuracy
        label_u = batch_u['label']

        input_x = input_x.to(self.device)
        input_x2 = input_x2.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)
        input_u2 = input_u2.to(self.device)
        label_u = label_u.to(self.device)

        return input_x, input_x2, label_x, input_u, input_u2, label_u
