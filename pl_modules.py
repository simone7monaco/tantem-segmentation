import torch

from utils import object_from_dict

from PIL import Image
import pytorch_lightning as pl
import torchmetrics as tm

import numpy as np
from copy import deepcopy

from losses import BinaryFocalLoss, AggregationLoss
from utils import Patcher, save_nii


class SegmentModel(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = self.hparams.model.get('name', '').lower()
        self.model = object_from_dict(self.hparams.model)
        self.loss = object_from_dict(self.hparams.loss)

        train_metrics = torch.nn.ModuleDict({
            'iou': tm.JaccardIndex(task='binary'),
            'precision': tm.Precision(task='binary'),
            'recall': tm.Recall(task='binary'),
            # 'dice': tm.F1Score(task='binary'),
            'pdice': tm.F1Score(task='binary', average='samples'),
        })
        self.metrics = torch.nn.ModuleDict({
            '_train': train_metrics,
            '_val': deepcopy(train_metrics),
            '_test': deepcopy(train_metrics),
        })

    def forward(self, batch: torch.Tensor, masks: torch.Tensor=None, iw: torch.Tensor=None) -> torch.Tensor:
        logits = self.model(batch)
        if masks is not None:
            if self.hparams.iw and iw is not None:
                loss = self.loss(logits, masks, iw)
            else:
                loss = self.loss(logits, masks)
        else:
            loss = None
        logits_ = (logits.detach() > self.hparams.test_parameters['threshold'])
        return logits_, loss

    def compute_loss(self, losses):
        return losses
    
    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams.optimizer,
            params=self.parameters(),
        )
        self.optimizers = [optimizer]
        if hasattr(self.hparams, 'scheduler'):
            scheduler = object_from_dict(self.hparams.scheduler, optimizer=optimizer)
            return [optimizer], [scheduler]
        return optimizer
    
    def training_step(self, batch, batch_idx):
        features = batch["features"]
        masks = batch["masks"]
        iw = batch.get("iw", None)

        logits_, loss = self.forward(features, masks, iw=iw)

        if isinstance(loss, dict):
            self.log_dict({f"train/{k}": v for k, v in loss.items()})
        loss = self.compute_loss(loss)

        for metric_name, metric in self.metrics['_train'].items():
            metric(logits_, masks.int())
            self.log(f"train/{metric_name}", metric, prog_bar=True)
        self.log("train/loss", loss)
        self.log("lr", self._get_current_lr())
        return {"loss": loss}

    def _get_current_lr(self):
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]
        return lr

    def validation_step(self, batch, batch_id):
        features = batch["features"]
        masks = batch["masks"]
        
        logits_, _ = self.forward(features)

        for metric_name, metric in self.metrics['_val'].items():
            metric(logits_, masks.int())
            self.log(f"val/{metric_name}", metric)
        self.eval_extra_metrics(logits_, features, masks, split='val')

    def on_train_epoch_end(self):
        self.log("epoch", float(self.trainer.current_epoch))

    def test_step(self, batch, batch_id):
        features, masks = batch["features"], batch["masks"]
        logits_, _ = self.forward(features)

        for i in range(features.shape[0]):
            name = batch["image_id"][i]
            ilogits_ = logits_[i].squeeze().cpu().numpy()

            file_path = self.hparams.callbacks['checkpoint_callback']['dirpath'] /'result'/'test'
            if len(features.shape) > 4:
                save_nii(file_path/f"{name}.nii", ilogits_.astype(np.float32))
            else:            
                Image.fromarray(ilogits_.astype(np.uint8)*255).save(file_path/f"{name}.png")

        for metric_name, metric in self.metrics['_test'].items():
            metric(logits_, masks.int())
            self.log(f"test/{metric_name}", metric)
        self.eval_extra_metrics(logits_, features, masks, split='test')

    def eval_extra_metrics(self, logits_, batch, masks, split):
        pass


class TandemSegmentModel(SegmentModel):
    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.classif_head = object_from_dict(self.hparams.classifier)

        self.classif_loss = BinaryFocalLoss(alpha=.1, gamma=2)
        if self.hparams.apply_patches:
            self.aggregation_loss = AggregationLoss()

        self.patcher = Patcher(kernel=self.hparams.patch_size)

        extra_metrics = torch.nn.ModuleDict({
            'classif_iou': tm.JaccardIndex(task='binary'),
            'classif_precision': tm.Precision(task='binary'),
            'classif_recall': tm.Recall(task='binary'),
        })
        self.extra_metrics = torch.nn.ModuleDict({
            '_train': extra_metrics,
            '_val': deepcopy(extra_metrics),
            '_test': deepcopy(extra_metrics),
        })

    def forward(self, batch: torch.Tensor, masks: torch.Tensor=None, iw: torch.Tensor=None, return_tandem: bool=False) -> torch.Tensor:
        logits = self.model(batch)
        if masks is not None:
            if self.hparams.iw:
                seg_loss = self.loss(logits, masks, iw)
            else:
                seg_loss = self.loss(logits, masks)
        
        logits_ = (logits.detach() > self.hparams.test_parameters['threshold'])
        classif_input, cl_labels = self.extract_patches(logits_, batch, gt=masks)
        classif_logits = self.classif_head(classif_input)
        if masks is not None:
            classif_loss = self.classif_loss(classif_logits.squeeze(), cl_labels)
            losses = {"seg_loss": seg_loss, "classif_loss": classif_loss}
        else:
            losses = None
        
        if hasattr(self.hparams, 'apply_patches') and self.hparams.apply_patches: # TODO: Ensure the softmask is doing something decent
            logits, class_mask = self.apply_patch_softmask(logits, classif_logits)
            logits_ = (logits > self.hparams.test_parameters['threshold'])
            if masks is not None:
                losses["aggr_loss"] = self.aggregation_loss(class_mask)
        if return_tandem:
            if masks is not None:
                return logits_, losses, classif_logits
            else:
                return logits_, classif_logits
        return logits_, losses

    def get_preds(self, x, get_logits=True):
        logits = self.model(x)
        logits_ = (logits.detach() > self.hparams.test_parameters['threshold'])
        classif_input, _ = self.extract_patches(logits_, x)
        cl_pred = self.classif_head(classif_input)
        cl_pred = torch.tile(cl_pred.unsqueeze(2).unsqueeze(3), (1, 1, self.patcher.kernel, self.patcher.kernel)) # shape (B, 1, H, W)
        patch_mask = self.patcher.depatch(cl_pred)
        patch_mask = patch_mask.tanh().clip(0)
        if get_logits:
            return logits, patch_mask
        return logits_, patch_mask
    
    def compute_loss(self, losses):
        loss = losses["seg_loss"] + losses["classif_loss"] * self.hparams.tandem
        if self.hparams.apply_patches:
            loss += losses["aggr_loss"] * self.hparams.apply_patches
        return loss
    
    def extract_patches(self, pred_bool, image, cutoff=10, iou_thr=0.4, gt=None):
        """
        Efficient computation of the class labels for each patch.
        If gt is not None, the function associate a label to each patch as follows:
         - 0: negative patch (the prediction finds a cyst where there is none OR the prediction doesn't find a cyst where there is one)
         - 1: positive patch (the prediction finds a cyst where there is one) 
        A cyst is present if the GT mask has a iou > cutoff with the prediction.
        ==========
        Returns
        - im_patches: tensor of shape (N*B, 3, p_size, p_size)
        """
        
        im_patches = self.patcher.patch(image)
        pred_patches = self.patcher.patch(pred_bool).bool()
        assert torch.all(self.patcher.depatch(im_patches) == image)
        assert torch.all(self.patcher.depatch(pred_patches.float()) == pred_bool)

        classif_input = torch.cat([im_patches, pred_patches], dim=1)
        if gt is not None:
            gt_patches = self.patcher.patch(gt)
            persample_dims = tuple(np.arange(len(gt_patches.size()))[1:])
            # set to 0 all the patches having sum < cutoff, keeping dimension (N*B, 1, p_size, p_size)
            gt_patches = torch.where(torch.tile(gt_patches.sum(dim=persample_dims, keepdim=True).lt(cutoff), 
                                                [1]+list(gt_patches.shape[1:])),
                                     torch.zeros_like(gt_patches), gt_patches).bool()

            iou = ((gt_patches & pred_patches).sum(dim=persample_dims).float() + 1e-6) / ((gt_patches | pred_patches).sum(dim=persample_dims).float() + 1e-6)
            cl_labels = (iou > iou_thr).float()
            return classif_input, cl_labels
        return classif_input, None
    
    def apply_patch_softmask(self, batch, cl_pred):
        """
        batch: images of size (B, 3, H, W)
        cl_pred: classifier predictions of size (B*N, 1), it is the output of the classif_head passed through an activation to get the probability of considering a patch or not
        """
        cl_pred = torch.tile(cl_pred.unsqueeze(2).unsqueeze(3), (1, 1, self.patcher.kernel, self.patcher.kernel)) # shape (B, 1, H, W)
        patch_mask = self.patcher.depatch(cl_pred) # shape (B, 1, H, W)
        return batch * patch_mask.tanh().clip(0), patch_mask
    
    def eval_extra_metrics(self, logits_, features, masks, split='_test'):
        classif_input, cl_labels = self.extract_patches(logits_, features, gt=masks)
        classif_logits = self.classif_head(classif_input)
        classif_logits_ = classif_logits.squeeze().tanh().clip(0).round()

        for metric_name, metric in self.extra_metrics[f"_{split}"].items():
            metric(classif_logits_, cl_labels)
            self.log(f"{split}/{metric_name}", metric)