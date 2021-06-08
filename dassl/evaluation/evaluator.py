import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import confusion_matrix

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100. * self._correct / self._total
        err = 100. - acc
        results['accuracy'] = acc
        results['error_rate'] = err

        print(
            '=> result\n'
            '* total: {:,}\n'
            '* correct: {:,}\n'
            '* accuracy: {:.2f}%\n'
            '* error: {:.2f}%'.format(self._total, self._correct, acc, err)
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print('=> per-class result')
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100. * correct / total
                accs.append(acc)
                print(
                    '* class: {} ({})\t'
                    'total: {:,}\t'
                    'correct: {:,}\t'
                    'acc: {:.2f}%'.format(
                        label, classname, total, correct, acc
                    )
                )
            print('* average: {:.2f}%'.format(np.mean(accs)))

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize='true'
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, 'cmat.pt')
            torch.save(cmat, save_path)
            print('Confusion matrix is saved to "{}"'.format(save_path))

        return results
