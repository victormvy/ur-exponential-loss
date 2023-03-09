import numpy as np
import torch
from scipy.special import softmax
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.nn.functional import softmax as torch_softmax


class UnimodalCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, num_classes: int = 5, eta: float = 0.85, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.eta = eta

        # Default class probs initialized to ones
        self.register_buffer('cls_probs', torch.ones(
            num_classes, num_classes).float())

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        y_prob = self.get_buffer('cls_probs')[target]
        target_oh = one_hot(target, self.num_classes)

        y_true = (1.0 - self.eta) * target_oh + self.eta * y_prob

        return super().forward(input, y_true)

def get_exponential_probabilities(n, p = 1.0):
	probs = []

	for true_class in range(0, n):
		probs.append(-np.abs(np.arange(0, n) - true_class)**p)

	return softmax(np.array(probs), axis=1)

class ExponentialCrossEntropyLoss(UnimodalCrossEntropyLoss):
    def __init__(self, num_classes: int = 5, eta: float = 0.85, p = 1.0, **kwargs):
        super().__init__(num_classes, eta, **kwargs)

        if p < 1.0 or p > 2.0:
            raise ValueError('p must be in the range [1.0, 2.0]')

        self.p = p

        self.cls_probs = torch.tensor(
            get_exponential_probabilities(num_classes, p=self.p)).float()


if __name__ == '__main__':
    n_classes = 5
    targets = torch.randint(0, n_classes, (1000,))
    targets_oh = one_hot(targets, n_classes)
    noise = torch.rand(1000, n_classes)
    probs = torch_softmax(targets_oh + noise, dim=1)

    cce_loss_fn = CrossEntropyLoss()
    cce_loss_val = cce_loss_fn(probs, targets).numpy()

    print('CrossEntropyLoss: ', cce_loss_val)

    for p in np.linspace(1.0, 2.0, 11):
        exp_loss_fn = ExponentialCrossEntropyLoss(num_classes = n_classes, p = p)
        exp_loss_val = exp_loss_fn(probs, targets).numpy()

        print(f'ExponentialCrossEntropyLoss (p = {p:.2f}): {exp_loss_val:.10f}')
    