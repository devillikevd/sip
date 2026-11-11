import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from PIL import Image
from torch.utils.data import dataset
import torchvision.transforms as T

VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_image_files(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
    files = sorted(
        f for f in os.listdir(directory) if f.lower().endswith(VALID_IMAGE_EXTENSIONS)
    )
    if not files:
        raise ValueError(f"No image files found in directory: {directory}")
    return [os.path.join(directory, f) for f in files]


class LoLDataset(dataset.Dataset):
    """Paired low-light / normal-light dataset loader for LOL-style directory layouts.

    Preserves the baseline MIRNet normalization `[0, 1]`
    and default 64x64 center-crop behavior while adding sorted pairing, path validation,
    and optional `resize` / `full` transform modes for baseline verification and training.
    """

    def __init__(self, low_light_root, target_root, img_size=64, mode="crop"):
        super().__init__()
        self.low_light_root = low_light_root
        self.target_root = target_root
        self.img_size = img_size
        self.mode = mode

        self.lol_fnames = _list_image_files(low_light_root)
        self.target_fnames = _list_image_files(target_root)

        if len(self.lol_fnames) != len(self.target_fnames):
            raise ValueError(
                f"Mismatched dataset counts: {len(self.lol_fnames)} low-light images in "
                f"{low_light_root} vs {len(self.target_fnames)} target images in {target_root}"
            )

        transforms_list = []
        if img_size is not None and mode == "crop":
            transforms_list.append(T.CenterCrop((img_size, img_size)))
        elif img_size is not None and mode == "resize":
            transforms_list.append(T.Resize((img_size, img_size)))
        elif mode not in ("crop", "resize", "full", None):
            raise ValueError(f"Unsupported transform mode: {mode}")

        transforms_list.extend(
            [
                T.ToTensor(),
                T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            ]
        )
        self.transform = T.Compose(transforms_list)

    def __getitem__(self, idx):
        lol = Image.open(self.lol_fnames[idx]).convert("RGB")
        target = Image.open(self.target_fnames[idx]).convert("RGB")
        lol = self.transform(lol)
        target = self.transform(target)
        return lol, target

    def __len__(self):
        return len(self.lol_fnames)


class PairedRandomPatchDataset(dataset.Dataset):
    """Paired dataset with synchronized random patch cropping and flip/rotate augmentation.

    Used by the Phase 6 hybrid training pipeline (`train_hybrid.py`) for training,
    while validation uses deterministic full/resized evaluation via `LoLDataset`.
    """

    def __init__(
        self,
        low_light_root,
        target_root,
        patch_size=128,
        augment=True,
    ):
        super().__init__()
        self.lol_fnames = _list_image_files(low_light_root)
        self.target_fnames = _list_image_files(target_root)
        if len(self.lol_fnames) != len(self.target_fnames):
            raise ValueError(
                f"Mismatched dataset counts: {len(self.lol_fnames)} vs {len(self.target_fnames)}"
            )
        self.patch_size = patch_size
        self.augment = augment
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.lol_fnames)

    def __getitem__(self, idx):
        import random

        low = Image.open(self.lol_fnames[idx]).convert("RGB")
        high = Image.open(self.target_fnames[idx]).convert("RGB")

        if self.patch_size is not None:
            w, h = low.size
            if w < self.patch_size or h < self.patch_size:
                low = low.resize(
                    (max(w, self.patch_size), max(h, self.patch_size)), Image.BICUBIC
                )
                high = high.resize(
                    (max(w, self.patch_size), max(h, self.patch_size)), Image.BICUBIC
                )
                w, h = low.size
            x0 = random.randint(0, w - self.patch_size)
            y0 = random.randint(0, h - self.patch_size)
            box = (x0, y0, x0 + self.patch_size, y0 + self.patch_size)
            low = low.crop(box)
            high = high.crop(box)

        if self.augment:
            if random.random() < 0.5:
                low = low.transpose(Image.FLIP_LEFT_RIGHT)
                high = high.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                low = low.transpose(Image.FLIP_TOP_BOTTOM)
                high = high.transpose(Image.FLIP_TOP_BOTTOM)

        return self.to_tensor(low), self.to_tensor(high)
