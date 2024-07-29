import argparse
import yaml
import wandb

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pl_modules import SegmentModel, TandemSegmentModel
from dataloaders import CystDataModule, TumorDataModule
from utils import *


def get_args():
    parser = argparse.ArgumentParser(description='CV with selected experiment as test set and train/val (+test) stratified from the others')
    parser.add_argument("-d", "--dataset", type=str, help="Select dataset to which apply tandem. Can be either cyst (2D) or tumor (3D)", default='cyst', choices=['cyst', 'tumor'])
    parser.add_argument("-c", "--config_path", type=Path, help="Custom config path. If empty, default is taken for the associated dataset", default=None)
    parser.add_argument("--model_style", type=str, default=None, choices=['baseline', 'tandem', 'tandem-applied'])
    parser.add_argument('--tandem', type=float, default=0, help = "Apply the tandem training strategy if != 0. This will be the coefficient for the classif loss.")
    parser.add_argument('--apply_patches', type=float, default=0, help = "Apply the patching strategy masking if != 0. This will be the loefficient of the 3rd loss contribution.")
    parser.add_argument("-s", "--seed", type=int, default=7, help="Change the seed to the desired one.")
    
    parser.add_argument("--tag", type=str, help="Add custom tag on the wandb run (only one tag is supported).", default=None)
    parser.add_argument('--gpu', type=int, default=None, help = "Select the GPU to use. If none take the less used one.")
    parser.add_argument('--force_retrain', nargs='?', type=str2bool, default=False, const=True, help = "Delete everything in the folder.")
    parser.add_argument('--wb', nargs='?', type=str2bool, default=False, const=True, help = "Prevent Wandb to save validation result for each step.")    
    return parser.parse_args()

def main(args):
    PLModel = TandemSegmentModel if args.tandem != 0 else SegmentModel
    DataModule = CystDataModule if args.dataset == 'cyst' else TumorDataModule
    if args.config_path is None:
        args.config_path = Path(f"configs/base_{args.dataset}.yaml")

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    args_to_hparams = ["tandem", "apply_patches", "dataset", "seed"]
    name = f"{args.dataset}_" + "_".join([f"{hp}_{getattr(args, hp)}" for hp in args_to_hparams if getattr(args, hp, None) is not None])
    
    torch.set_float32_matmul_precision('medium')
    wandb.init(project="rene-policistico-cyst_segmentation",
            tags=[args.tag] if args.tag else None, reinit=True,
            name=None
            ) if args.wb else None
    hparams = init_training(args, hparams, name,
                            args_to_hparams=args_to_hparams,
                            )

    if args.force_retrain:
        restore_folder(hparams["callbacks"]["checkpoint_callback"]["dirpath"])

    callbacks = []
    for cb_hparams in hparams["callbacks"].values():
        callbacks.append(object_from_dict(cb_hparams))

    logger = WandbLogger() if args.wb else None
    if logger:
        logger.log_hyperparams(hparams)
        # logger.watch(model, log='all', log_freq=1)

    data = DataModule(**dict(hparams, **args.__dict__))
    model = PLModel(**hparams)

    if getattr(args, 'seed', None) is not None:
        pl.seed_everything(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.gpu is not None:
        devices = [args.gpu]
    else:
        devices = get_gpu_memory_map()
        devices = [min(devices, key=devices.get)]
        print(f"\n🆓  Using GPU {devices} (automatically detected)\n")

    trainer = pl.Trainer(
        accelerator='gpu', devices=devices, max_epochs=hparams['train_parameters']['epochs'],
        callbacks=callbacks, logger=logger,
        gradient_clip_val=5.0, num_sanity_val_steps=0, precision="bf16-mixed",
        log_every_n_steps=5
    )
    success = hparams["callbacks"]["checkpoint_callback"]["dirpath"] / ".success"
    if not success.exists():
        trainer.fit(model, datamodule=data)
        print("\nTraining completed\n")
        success.touch()

    test_preds = hparams["callbacks"]["checkpoint_callback"]["dirpath"]/'result'/'test'
    test_preds.mkdir(exist_ok=True, parents=True)
    
    if not any(test_preds.glob('*png')):
        print(f'Starting evaluation on test set ({len(data.test_dataloader().dataset)} samples)')
        print(f"Results will be saved in '{test_preds}'\n")
        model = PLModel.load_from_checkpoint(next(hparams["callbacks"]["checkpoint_callback"]["dirpath"].glob("*.ckpt")))
        trainer.test(model, datamodule=data)
    else:
        print(f"\nTest already completed (results in '{test_preds}'")

    if wandb.run is not None:
        wandb.finish()
    return


if __name__ == '__main__':
    args = get_args()

    if args.model_style == 'tandem':
        args.tandem = 50.
    elif args.model_style == 'tandem-applied':
        args.tandem = 50.
        args.apply_patches = 1.
    main(args)
