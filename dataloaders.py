from pathlib import Path
from typing import List, Dict, Any, Tuple

import nibabel
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_toolbelt.utils.torch_utils import image_to_tensor
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler

import albumentations as albu

from skimage.io import imread
from skimage import measure
from utils import get_samples, simplify_names, date_to_exp
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold

train_transform = albu.Compose(
    [
        # albu.LongestMaxSize(max_size=512, always_apply=False, p=1),
        albu.HorizontalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.RandomBrightnessContrast(p=0.5),
        albu.RandomGamma(p=0.5),
        albu.Normalize(max_pixel_value=255, always_apply=False, p=1.0,
                       mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
test_transform = albu.Compose(
    [
        # albu.LongestMaxSize(max_size=512, always_apply=False, p=1),
        albu.Normalize(max_pixel_value=255, always_apply=False, p=1.0,
                       mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def split_cyst_dataset(hparams):
    samples = get_samples(hparams["data_path"]/'images', hparams["data_path"]/'masks')
    
    names = [file[0].stem for file in samples]
    unpack = [simplify_names(name) for name in names]
    df = pd.DataFrame({
        "filename": names,
        "treatment": [u[1] for u in unpack],
        "exp": [date_to_exp(u[0]) for u in unpack],
        "tube": [u[2] for u in unpack],
    })
    df["te"] = (df.treatment + '_' + df.exp.astype(str) + '_' + df.tube.astype(str)).astype('category')
    
    test_tubes = df.groupby('treatment').te.first()[:3].values
    test_idx = df[df.te.isin(test_tubes)].index
    
    test_samp = [x for i, x in enumerate(samples) if i in test_idx]
    samples = [x for i, x in enumerate(samples) if i not in test_idx]
    df = df.drop(test_idx)
        
    df, samples = shuffle(df, samples, random_state=hparams["seed"])
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = list(gkf.split(df.filename, groups=df.te))[0]
    
    train_samp = [tuple(x) for x in np.array(samples)[train_idx]]
    val_samp = [tuple(x) for x in np.array(samples)[val_idx]]
    
    return {
        "train": train_samp,
        "valid": val_samp,
        "test": test_samp
    }


class CystSegDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, Path]],
        transform: albu.Compose,
        noG=False,
        create_iw=False,
    ):
        self.samples = samples
        self.transform = transform
        self.length = len(self.samples)
        self.noG = noG
        self.create_iw = create_iw

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path, mask_path = self.samples[idx]
        image = imread(image_path)
        mask = imread(mask_path)

        if self.noG:
            image[:, :, 1] = 0

        # apply augmentations
        if self.transform is not None:
            sample = self.transform(image=image, mask=mask.astype(np.uint8))
            image, mask = sample["image"], sample["mask"]
        mask = (mask > 0).astype(np.uint8)

        res = {
            "image_id": image_path.stem,
            "features": image_to_tensor(image),
            "masks": torch.from_numpy(mask).unsqueeze(0).float(),
        }
        if self.create_iw:
            res["iw"] = torch.from_numpy(self.get_iw(mask)).unsqueeze(0).float()
        return res

    @staticmethod
    def get_iw(y_bin: np.ndarray, w_min: float = 1., w_max: float = 2e5):
        """
        Inverse weighting for class imbalance, 
        based on the connected as in https://arxiv.org/pdf/2007.10033
        """
        cc = measure.label(y_bin, connectivity=2)
        weight = np.zeros_like(cc, dtype='float32')
        cc_items = np.unique(cc)
        K = len(cc_items) - 1
        N = np.prod(cc.shape)
        for i in cc_items:
            weight[cc == i] = N / ((K + 1) * np.sum(cc == i))
        return np.clip(weight, w_min, w_max)


class CystDataModule(pl.LightningDataModule):
    def __init__(self, verbose=False, **hparams):
        super().__init__()
        
        for k in ['model', 'optimizer', 'loss']:
            if k in hparams:
                hparams.pop(k)
        self.save_hyperparameters(logger=False)
        self.verbose = verbose

        if not self.hparams['data_path'].exists():
            raise ValueError(f"Data path {self.hparams['data_path']} does not exist.")

        splits = split_cyst_dataset(self.hparams)
        self.train_samples=splits['train']
        self.val_samples=splits['valid']
        self.test_samples=splits['test']

        if self.verbose:
            print("Len train samples = ", len(self.train_samples))
            print("Len val samples = ", len(self.val_samples))

        self.train_aug = train_transform
        self.val_aug = test_transform
        self.test_aug = test_transform

        self.batch_size = self.hparams.train_parameters["batch_size"]
        self.val_batch_size = self.hparams.val_parameters["batch_size"]

    def train_dataloader(self):
        result = DataLoader(
            CystSegDataset(self.train_samples, self.train_aug, noG=self.hparams["noG_preprocessing"], create_iw=self.hparams["iw"]),
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
        )
        if self.verbose:
            print("Train dataloader = ", len(result))
        return result

    def val_dataloader(self):
        result = DataLoader(
            CystSegDataset(self.val_samples, self.val_aug, noG=self.hparams["noG_preprocessing"]),
            batch_size=self.val_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
        )
        if self.verbose:
            print("Val dataloader = ", len(result))
        return result

    def test_dataloader(self):
        result = DataLoader(
            CystSegDataset(self.test_samples, self.test_aug, noG=self.hparams["noG_preprocessing"]),
            batch_size=self.val_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
        ) if self.test_samples is not None else []
        if self.verbose:
            print("Test dataloader = ", len(result))
        return result


def split_tumor_dataset(data_path):
    df = pd.read_csv(data_path/'metadata.csv', index_col='id')
    return {
        "train": df[df.n_batch.eq(2)].index,
        "val": df[df.n_batch.eq(1)].index,
        "test": df[df.n_batch.eq(1)].index
    }


class LITS(Dataset):
    def __init__(self, data_path, metadata_rpath='metadata.csv', 
                 indices=None, create_iw=False, image_size=128):
        super().__init__()
        self.data_path = data_path
        self.df = pd.read_csv(data_path/metadata_rpath, index_col='id')
        if indices is not None:
            self.df = self.df.loc[indices]
        self.n_chans_image = 1 # one modality used (one single channel)
        self.create_iw = create_iw
        self.image_size = image_size
    
    def __len__(self) -> int:
        return self.df.shape[0]
    
    def __getitem__(self, index) -> Any:
        ct = self.load_nii(self.data_path/self.df['CT'].iloc[index])
        target = self.load_nii(self.data_path/self.df['target'].iloc[index])

        ct = self.scale_ct(ct)

        if ct.shape[2] < self.image_size:
            ct = np.pad(ct, ((0, 0), (0, 0), (0, self.image_size - ct.shape[2])))       
            target = np.pad(target, ((0, 0), (0, 0), (0, self.image_size - target.shape[2])))
        cc = self.get_cc(target)

        if cc.any():
            tumor_centers = self.load_tumor_centers(cc) # list of arrays of size n_pixels x 3, we want to average and get the center for each tumor, then average again to get the center of all tumors
            tumor_centers = np.array([np.mean(tumor_center, axis=0) for tumor_center in tumor_centers])
            center = np.mean(tumor_centers, axis=0).astype(int)
        else:
            center = np.array(ct.shape) // 2
        ct, target, cc = self.center_crop([ct, target, cc], center)

        assert ct.shape == (self.image_size, self.image_size, self.image_size)
        target = (target > 0).astype(np.uint8)

        # n_tumors = self.df.n_tumors.iloc[index]
        res =  {
            'image_id': index,
            'features': torch.from_numpy(ct).unsqueeze(0).float(),
            'masks': torch.from_numpy(target).unsqueeze(0).float(),
            # 'n_tumors': n_tumors,
        }
        if self.create_iw:
            res["iw"] = torch.from_numpy(self.get_iw(cc)).unsqueeze(0).float()
        return res

    def center_crop(self, arrays, center):
        def crop(array, center):
            start = np.maximum(center - self.image_size // 2, 0)
            end = np.minimum(center + self.image_size // 2, array.shape)
            if np.any(end - start < self.image_size):
                start[end - start < self.image_size] = np.maximum(end[end - start < self.image_size] - self.image_size, 0)
            if np.any(end - start < self.image_size):
                end[end - start < self.image_size] = np.minimum(start[end - start < self.image_size] + self.image_size, np.array(array.shape)[end - start < self.image_size])
            return array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        
        return [crop(array, center) for array in arrays]

    @staticmethod
    def load_nii(filepath, **kwargs):
        return np.array(nibabel.load(str(filepath), **kwargs).get_fdata())
    
    @staticmethod
    def scale_ct(x: np.ndarray, min_value: float = -300, max_value: float = 300) -> np.ndarray:
        x = np.clip(x, a_min=min_value, a_max=max_value).astype(np.float32)
        x -= x.min()
        x /= x.max()
        return x
    
    @staticmethod
    def get_cc(y_bin):
        return measure.label(y_bin, connectivity=3)

    @staticmethod
    def load_tumor_centers(cc):
        n_labels = np.max(cc)        
        return [np.argwhere(cc == label) for label in range(1, n_labels + 1)]
    
    @staticmethod
    def get_iw(cc, w_min=1., w_max=2e5):
        weight = np.zeros_like(cc, dtype='float32')
        cc_items = np.unique(cc)
        K = len(cc_items) - 1
        N = np.prod(cc.shape)
        for i in cc_items:
            weight[cc == i] = N / ((K + 1) * np.sum(cc == i))
        return np.clip(weight, w_min, w_max)


class PatientSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.patient_sampling_weights = dataset.n_tumors / dataset.n_tumors.sum()
        self.indices = np.arange(len(self.dataset))
        
    def __iter__(self):
        return iter(np.random.choice(self.indices, size=len(self.indices), 
                                     p=self.patient_sampling_weights))
    
    def __len__(self):
        return len(self.dataset)
    

class TumorDataModule(pl.LightningDataModule):
    def __init__(self, data_path, metadata_rpath='metadata.csv', 
                 sample_train=True, **hparams):
        super().__init__()
        for k in ['model', 'optimizer', 'loss']:
            if k in hparams:
                hparams.pop(k)
        self.save_hyperparameters(logger=False)
        self.data_path = data_path
        self.metadata_rpath = metadata_rpath
        self.sample_train = sample_train
        self.indices = split_tumor_dataset(data_path)
        self.batch_size = self.hparams.train_parameters["batch_size"]
    
    def train_dataloader(self):
        dataset = LITS(self.data_path, self.metadata_rpath, 
                       self.indices['train'], create_iw=self.hparams["iw"])
        if self.sample_train:
            weights = dataset.df.n_tumors / dataset.df.n_tumors.sum()
            sampler = WeightedRandomSampler(weights.values, num_samples=int(2*len(dataset)), replacement=True)
            # sampler = PatientSampler(dataset.df)
            return DataLoader(dataset, batch_size=self.batch_size, 
                            sampler=sampler, num_workers=self.hparams.num_workers)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.hparams.num_workers)
    
    def val_dataloader(self):
        return DataLoader(LITS(self.data_path, self.metadata_rpath, self.indices['val']), 
                          batch_size=self.batch_size, num_workers=self.hparams.num_workers)
    
    def test_dataloader(self):
        return DataLoader(LITS(self.data_path, self.metadata_rpath, self.indices['test']), 
                          batch_size=self.batch_size, num_workers=self.hparams.num_workers)