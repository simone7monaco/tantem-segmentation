import numpy as np
import torch
from dataloaders import CystSegDataset, LITS, split_tumor_dataset
from skimage.io import imread
from utils import get_id2_file_paths
from skimage import measure


delims = np.array([55, 235])
delims = (72/25)**2 * delims # to px

size_delimiters = {
    'cyst': delims,
    'tumor': np.array([200, 2000])
}

def get_samples(paths):
    im2paths = [get_id2_file_paths(path) for path in paths]
    keys = set(im2paths[0].keys())
    for i in range(1, len(im2paths)):
        keys = keys.intersection(im2paths[i].keys())
    return [tuple((path[file_id] for path in im2paths)) for file_id in keys]

class EvalCystDataset(CystSegDataset):
    def __init__(self, data_path, pred_path):
        samples = get_samples([data_path/'images', data_path/'masks', pred_path])
        super().__init__([(s[0], s[1]) for s in samples], 
                         transform=None, noG=True, create_iw=False)
        self.preds = [s[2] for s in samples]

    def __getitem__(self, idx):
        pred_path = self.preds[idx]
        pred = imread(pred_path)

        image_id, image, mask = super().__getitem__(idx).values()
        sizes = self.sizes_tensor(mask.squeeze().numpy())
        psizes = self.sizes_tensor(pred)

        return {
            "image_id": image_id,
            "features": image,
            "masks": mask,
            "sizes": torch.from_numpy(sizes).unsqueeze(0).float(),
            "preds": torch.from_numpy(pred).unsqueeze(0).float(),
            "psizes": torch.from_numpy(psizes).unsqueeze(0).float()
        }
    @staticmethod
    def sizes_tensor(tensor):
        _sizes = measure.label(tensor > 0)
        sizes = np.zeros_like(tensor, dtype=np.float32)
        for i in np.unique(_sizes):
            if i>0:
                sizes[_sizes == i] = np.sum(_sizes == i)
        return sizes

    
class EvalTumorDataset(LITS):
    def __init__(self, data_path, pred_path):
        indices = split_tumor_dataset(data_path)
        super().__init__(data_path, indices=indices['test'], create_iw=False)
        self.pred_path = pred_path

    def __getitem__(self, idx):
        image_id, image, mask = super().__getitem__(idx).values()
        pred = self.load_nii(self.pred_path / f'{image_id}.nii')

        sizes = self.sizes_tensor(mask)
        psizes = self.sizes_tensor(pred)

        return {
            "image_id": image_id,
            "features": image,
            "masks": mask,
            "sizes": torch.from_numpy(sizes).unsqueeze(0).float(),
            "preds": torch.from_numpy(pred).unsqueeze(0).float(),
            "psizes": torch.from_numpy(psizes).unsqueeze(0).float()
        }
    
    @staticmethod
    def sizes_tensor(tensor):
        tensor = np.squeeze(tensor)
        _sizes = measure.label(tensor > 0)
        sizes = np.zeros_like(tensor, dtype=np.float32)
        for i in np.unique(_sizes):
            if i>0:
                sizes[_sizes == i] = np.sum(_sizes == i)
        return sizes