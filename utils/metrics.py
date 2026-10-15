"""Dataset confusion matrix for K classes; background=0; union foreground surfaces."""
import torch
import numpy as np
from .surface_metrics import surface_scores


class SegmentationMetrics:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        self.surface_sums = np.zeros(3, dtype=np.float64)
        self.image_count = 0

    def update(self, logits, target):
        prediction = logits.argmax(1)
        k = self.num_classes
        if logits.shape[1] != k or target.min() < 0 or target.max() >= k:
            raise ValueError('Expected matching K-channel logits and class IDs 0..K-1')
        bins = (k * target + prediction).detach().reshape(-1).cpu()
        self.cm += torch.bincount(bins, minlength=k*k).reshape(k, k)
        for p, t in zip(prediction.detach().cpu().numpy(), target.detach().cpu().numpy()):
            self.surface_sums += surface_scores(p > 0, t > 0)
            self.image_count += 1

    def compute(self):
        cm = self.cm.double(); diag = cm.diag(); counts = cm.sum(1)
        predicted = cm.sum(0); union = counts + predicted - diag
        def ratio(a, b): return float(a / b) if b else None
        per_class = [dict(class_id=i, IoU=ratio(diag[i], union[i]),
                          Accuracy=ratio(diag[i], counts[i]),
                          Dice=ratio(2*diag[i], counts[i]+predicted[i]),
                          Precision=ratio(diag[i], predicted[i]),
                          Recall=ratio(diag[i], counts[i]), pixels=int(counts[i]))
                     for i in range(self.num_classes)]
        def mean(values):
            defined = [v for v in values if v is not None]
            return sum(defined)/len(defined) if defined else None
        tn = cm[0,0].item(); fp = cm[0,1:].sum().item()
        fn = cm[1:,0].sum().item(); tp = cm[1:,1:].sum().item()
        return {
            'Foreground IoU': ratio(tp, tp+fp+fn),
            'Background IoU': per_class[0]['IoU'],
            'Mean IoU': mean([v['IoU'] for v in per_class]),
            'Dice': mean([v['Dice'] for v in per_class[1:]]),
            'Precision': mean([v['Precision'] for v in per_class[1:]]),
            'Recall': mean([v['Recall'] for v in per_class[1:]]),
            'Specificity': ratio(tn, tn+fp),
            'Pixel Accuracy': ratio(diag.sum(), cm.sum()),
            'Mean Accuracy': mean([v['Accuracy'] for v in per_class]),
            'Frequency Weighted IoU': ratio(sum(counts[i] * (per_class[i]['IoU'] or 0.) for i in range(self.num_classes)), cm.sum()),
            'HD95': float(self.surface_sums[0]/self.image_count) if self.image_count else None,
            'ASSD': float(self.surface_sums[1]/self.image_count) if self.image_count else None,
            'Boundary F1': float(self.surface_sums[2]/self.image_count) if self.image_count else None,
            'boundary_scope': 'union of all nonbackground classes; tolerance=2 pixels',
            'dice_scope': 'macro foreground classes with defined denominator',
            'per_class': per_class, 'confusion_matrix': self.cm.tolist(),
        }
