"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images



import importlib
from pathlib import Path

def instantiate_from_config(config, params = None):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    if params is None:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))
    else:
        return get_obj_from_str(config["target"])(**params, **config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_ext_list(dir_path, ext_list=['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']):
    img_dir = Path(dir_path)

    img_files = []
    for ext in ext_list:
        file_generator = img_dir.glob(f"**/*.{ext}")
        img_files.extend([file for file in file_generator])
    img_files.sort()
    return img_files