from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss, BinarySoftF1Loss
import torch
from torch import nn
import torch.nn.functional as F


class ComboLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5], losses=["jaccard", "bce"], unbalance=None, iw=False):
        super(ComboLoss, self).__init__()
        self.iw = iw
        if iw:
            self.losses = nn.ModuleDict({
                "jaccard": iwJaccardLoss(), # binary mode and from_logits are hardcoded
                "bce": iwBCEWithLogitsLoss(),
                "focal": iwFocalLoss()
            })
        else:
            losses_dict = {
                "jaccard": JaccardLoss(mode="binary", from_logits=True),
                "f1": BinarySoftF1Loss(),
                "bce": nn.BCEWithLogitsLoss(),
                "focal": BinaryFocalLoss(),
            }

        self.losses = nn.ModuleDict({n: losses_dict.get(n) for n in losses})
        self.weights = weights

    def forward(self, logits, masks, iw=None):
        total_loss = 0
        for weight, loss in zip(self.weights, self.losses.values()):
            if self.iw:
                ls_mask = loss(logits, masks, iw)
            else:
                ls_mask = loss(logits, masks)
            total_loss += weight * ls_mask
        return total_loss


class AggregationLoss(torch.nn.Module):
    """
    Idea: the mask is a softmask (i.e. values in [0,1]) that is applied to the image to enhance the cysts.
    a good segmentation model would produce a prediction that should be never changed by the mask. Then the 
    mask should be always 1.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, mask):
        return torch.nn.functional.mse_loss(mask, torch.ones_like(mask))
    

class iwFocalLoss(nn.Module):
    def __init__(self, **kwargs):
        super(iwFocalLoss, self).__init__()
        self.loss = BinaryFocalLoss(reduction="none", **kwargs)

    def forward(self, logits, masks, iw):
        loss = iw * self.loss(logits, masks)
        return loss.mean()
    
class iwBCEWithLogitsLoss(nn.Module):
    def __init__(self, **kwargs):
        super(iwBCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction="none", **kwargs)

    def forward(self, logits, masks, iw):
        loss = iw * self.loss(logits, masks)
        return loss.mean()
    
class iwJaccardLoss(nn.Module):
    def __init__(self, **kwargs):
        super(iwJaccardLoss, self).__init__()
        self.loss = JaccardLoss(mode="binary", from_logits=True, **kwargs)

    @staticmethod
    def soft_jaccard_score(output: torch.Tensor, target: torch.Tensor, iw: torch.Tensor, 
                           smooth: float = 0.0, eps: float = 1e-7, dims=None):
        if dims is not None:
            intersection = torch.sum(output * target * iw, dim=dims)
            cardinality = torch.sum((output + target) * iw, dim=dims)
        else:
            intersection = torch.sum(output * target * iw)
            cardinality = torch.sum((output + target) * iw)

        union = cardinality - intersection
        jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
        return jaccard_score
    
    def forward(self, logits, masks, iw):
        """
        reimplementing pytorch_toolbelt JaccardLoss to include inverse weights

        :param y_pred: Nx1xHxW
        :param y_true: Nx1xHxW
        :param iw: Nx1xHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        dims = (0, 2)

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)
        iw = iw.view(bs, 1, -1)
        scores = self.soft_jaccard_score(y_pred, y_true.type(y_pred.dtype), iw, smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.float()

        if self.classes is not None:
            loss = loss[self.classes]
        return loss.mean()
