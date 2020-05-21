import torch
import torch.nn as nn

from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.engine.trainer import SimpleNet
from dassl.data.transforms import build_transform
from dassl.modeling.ops.utils import create_onehot


class Experts(nn.Module):

    def __init__(self, n_source, fdim, num_classes):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(fdim, num_classes) for _ in range(n_source)]
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, i, x):
        x = self.linears[i](x)
        x = self.softmax(x)
        return x


@TRAINER_REGISTRY.register()
class DAELDG(TrainerX):
    """Domain Adaptive Ensemble Learning.

    DG version: only use labeled source data.

    https://arxiv.org/abs/2003.07325.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        n_domain = cfg.DATALOADER.TRAIN_X.N_DOMAIN
        batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        if n_domain <= 0:
            n_domain = self.dm.num_source_domains
        self.split_batch = batch_size // n_domain
        self.n_domain = n_domain

        self.conf_thre = cfg.TRAINER.DAEL.CONF_THRE

    def check_cfg(self, cfg):
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == 'RandomDomainSampler'
        assert len(cfg.TRAINER.DAEL.STRONG_TRANSFORMS) > 0

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.DAEL.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        self.dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = self.dm.train_loader_x
        self.train_loader_u = self.dm.train_loader_u
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes

    def build_model(self):
        cfg = self.cfg

        print('Building F')
        self.F = SimpleNet(cfg, cfg.MODEL, 0)
        self.F.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model('F', self.F, self.optim_F, self.sched_F)
        fdim = self.F.fdim

        print('Building E')
        self.E = Experts(self.dm.num_source_domains, fdim, self.num_classes)
        self.E.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.E)))
        self.optim_E = build_optimizer(self.E, cfg.OPTIM)
        self.sched_E = build_lr_scheduler(self.optim_E, cfg.OPTIM)
        self.register_model('E', self.E, self.optim_E, self.sched_E)

    def forward_backward(self, batch):
        parsed_data = self.parse_batch_train(batch)
        input, input2, label, domain = parsed_data

        input = torch.split(input, self.split_batch, 0)
        input2 = torch.split(input2, self.split_batch, 0)
        label = torch.split(label, self.split_batch, 0)
        domain = torch.split(domain, self.split_batch, 0)
        domain = [d[0].item() for d in domain]

        loss_x = 0
        loss_cr = 0
        acc = 0

        feat = [self.F(x) for x in input]
        feat2 = [self.F(x) for x in input2]

        for feat_i, feat2_i, label_i, i in zip(feat, feat2, label, domain):
            cr_s = [j for j in domain if j != i]

            # Learning expert
            pred_i = self.E(i, feat_i)
            loss_x += (-label_i * torch.log(pred_i + 1e-5)).sum(1).mean()
            expert_label_i = pred_i.detach()
            acc += compute_accuracy(pred_i.detach(),
                                    label_i.max(1)[1])[0].item()

            # Consistency regularization
            cr_pred = []
            for j in cr_s:
                pred_j = self.E(j, feat2_i)
                pred_j = pred_j.unsqueeze(1)
                cr_pred.append(pred_j)
            cr_pred = torch.cat(cr_pred, 1)
            cr_pred = cr_pred.mean(1)
            loss_cr += ((cr_pred - expert_label_i)**2).sum(1).mean()

        loss_x /= self.n_domain
        loss_cr /= self.n_domain
        acc /= self.n_domain

        loss = 0
        loss += loss_x
        loss += loss_cr
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss_x': loss_x.item(),
            'acc': acc,
            'loss_cr': loss_cr.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        input2 = batch['img2']
        label = batch['label']
        domain = batch['domain']

        label = create_onehot(label, self.num_classes)

        input = input.to(self.device)
        input2 = input2.to(self.device)
        label = label.to(self.device)

        return input, input2, label, domain

    def model_inference(self, input):
        f = self.F(input)
        p = []
        for k in range(self.dm.num_source_domains):
            p_k = self.E(k, f)
            p_k = p_k.unsqueeze(1)
            p.append(p_k)
        p = torch.cat(p, 1)
        p = p.mean(1)
        return p
