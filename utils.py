import argparse
import shutil
import subprocess
import pydoc

from pathlib import Path
from typing import Union, Dict, List, Tuple

import numpy as np
import nibabel as nb
import torch
import torch.nn.functional as F
import cv2
import re



def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return {i: memory for i, memory in enumerate(gpu_memory)}


def get_id2_file_paths(path: Union[str, Path]) -> Dict[str, Path]:
    return {x.stem: x for x in Path(path).glob("*.*")}


def restore_folder(path: Path) -> None:
    print(">> delete everything in the folder")
    for f in path.glob("*"):
        if f.is_dir():
            shutil.rmtree(f)
        else:
            f.unlink()
                    

def date_to_exp(date: str) -> int:
    date_exps = {'0919': 1, '1019': 2, '072020':3, '092020':3, '122020':4, '0721':5}
    date = ''.join((date).split('.')[1:])
    return date_exps[date]


# all_treats = {'ctrl', 'treat_1', 'treat_2', 'treat_3', 'treat_4', 'treat_5', 'treat_6', 'treat_7', 'treat_8', 'treat_9', 'treat_10'}
all_treats = {'ctrl', 't3', 'triac', 't4', 'tetrac', 'resv', 'dbd', 'lm609', 'uo', 'dbd+t4', 'uo+t4', 'lm609+t4', 'lm609+10ug.ml', 'lm609+2.5ug.ml'}

def simplify_names(filename: str) -> Tuple[str, str, str, str, str]:
    unpack = re.split(' {1,}_?|_', filename.strip())
    
    date_idx = [i for i, item in enumerate(unpack) if re.search('[0-9]{1,2}.[0-9]{1,2}.[0-9]{2,4}', item)][0]
    unpack = unpack[date_idx:]
    date = unpack[0]
    treatment = [x.upper() for x in unpack if x.lower() in all_treats][-1]

    side = [s for s in unpack if re.match('A|B', s)]
    side = side[0] if side else 'U'

    zstack = [s.lower() for s in unpack if re.match('[24]0[xX][0-9]{1,2}', s)][0]
    alt_zstack = [s for s in unpack if re.match('\([0-9]{1,}\)', s)]
    if alt_zstack: zstack = zstack.split('x')[0] + 'x' + alt_zstack[0][1:-1]
    z1, z2 = zstack.split('x')
    zstack = f"{z1}x{int(z2):02}"

    tube = [n for n in unpack if re.fullmatch('[0-9]*', n)][0]
    
    return date, treatment, tube, zstack, side


def str2bool(v: Union[str, bool]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_samples(image_path: Path, mask_path: Path) -> List[Tuple[Path, Path]]:
    """Couple masks and images.

    Args:
        image_path:
        mask_path:

    Returns:
    """
    image2path = get_id2_file_paths(image_path)
    mask2path = get_id2_file_paths(mask_path)
    return [(image_file_path, mask2path[file_id]) for file_id, image_file_path in image2path.items()]


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034
    return pydoc.locate(object_type)(**kwargs) if pydoc.locate(object_type) is not None else pydoc.locate(object_type.rsplit('.', 1)[0])(**kwargs)


def load_rgb(image_path: Union[Path, str]) -> np.array:
    """Load RGB image from path.
    Args:
        image_path: path to image
        lib: library used to read an image.
            currently supported `cv2` and `jpeg4py`
    Returns: 3 channel array with RGB image
    """
    if Path(image_path).is_file():
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    raise FileNotFoundError(f"File not found {image_path}")


def load_mask(path: Union[Path, str]) -> np.array:
    im = str(path)
    return (cv2.imread(im, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)


def save_nii(path_to_file, img):
    """Saves ``img`` of type `numpy.ndarray` in nifty format."""
    nb.save(nb.Nifti1Image(img, np.eye(4)), path_to_file)

#######################
####                  #
#### Training utils   #
####                  #
#######################

def init_training(args, hparams, name, args_to_hparams=[]):
    """
    Extracts arguments from argparse and adds them to hparams. Then converts all paths to Path objects.
    =================
    Args:
        args: argparse.Namespace
        hparams: dict
        name: str
        args_to_hparams: list of str
    """
    hparams["data_path"] = Path(hparams["data_path"])
    hparams["callbacks"]["checkpoint_callback"]["dirpath"] = Path(hparams["callbacks"]["checkpoint_callback"]["dirpath"])
    hparams["callbacks"]["checkpoint_callback"]["dirpath"] /= name
    hparams["callbacks"]["checkpoint_callback"]["dirpath"].mkdir(exist_ok=True, parents=True)

    for arg in args_to_hparams:
        if hasattr(args, arg):
            hparams[arg] = getattr(args, arg)    
    return hparams


class Patcher:
    """
    Class to extract patches from images and viceversa
    """
    def __init__(self, kernel: int=256):
        self.kernel = kernel
        self.stride = kernel # Non-overlapping patches

    def patch(self, images):
        C = images.size(1)
        self.dims = images.shape[2:]

        if len(self.dims) == 2:
            patches = images.unfold(2, self.kernel, self.stride).unfold(3, self.kernel, self.stride)
            patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, self.kernel, self.kernel)
        elif len(self.dims) == 3:
            patches = images.unfold(2, self.kernel, self.stride).unfold(3, self.kernel, self.stride).unfold(4, self.kernel, self.stride)
            patches = patches.contiguous().view(-1, C, self.kernel, self.kernel, self.kernel)
        else:
            raise ValueError("Only 2D and 3D images are supported")
        return patches  

    def depatch(self, patches):
        C = patches.size(1)
        D_out = [(D - self.kernel) // self.stride + 1 for D in self.dims]

        if len(self.dims) == 2:
            patches = patches.view(-1, D_out[0], D_out[1], C, self.kernel, self.kernel).permute(0, 3, 4, 5, 1, 2)
            patches = patches.contiguous().view(-1, C * self.kernel * self.kernel, np.prod(D_out))
            reconstructed = F.fold(patches, output_size=self.dims, kernel_size=self.kernel, stride=self.stride)
        else:
            patches = patches.view(-1, C, D_out[2], D_out[0], D_out[1], self.kernel, self.kernel, self.kernel)
            patches = patches.permute(0, 1, 5, 2, 6, 7, 3, 4)
            patches = patches.contiguous().view(-1, C * self.kernel * D_out[2] * self.kernel * self.kernel, D_out[0] * D_out[1])

            reconstructed = torch.nn.functional.fold(patches, 
                                        output_size=(self.dims[0], self.dims[1]),
                                        kernel_size=self.kernel,
                                        stride=self.stride)
            reconstructed = reconstructed.view(-1, C * self.kernel, D_out[2] * self.dims[0] * self.dims[1])

            reconstructed = torch.nn.functional.fold(reconstructed, 
                                        output_size=(self.dims[2], self.dims[0]*self.dims[1]),
                                        kernel_size=(self.kernel, 1), 
                                        stride=(self.stride, 1))
            reconstructed = reconstructed.view(-1, C, *self.dims)
        return reconstructed
    
    def check(self, images):
        patches = self.patch(images)
        output = self.depatch(patches)
        assert torch.allclose(images, output), "Patching and depatching not consistent"
        print("Patching and depatching consistent")