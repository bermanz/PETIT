from multiprocessing import cpu_count
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from torchvision.transforms import ToTensor, Compose, CenterCrop
from utils.deep import NetPhase
from configs import LwirChannel


def getDataLoader(
    is_train: bool, bandwidth: LwirChannel, batch_size: int, path_to_data: str = "data"
):
    dataset = Tau2AerialDS(Path(path_to_data), is_train=is_train, mono=bandwidth)
    print(f"{bandwidth} dataset was created")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=cpu_count(),
    )
    return dataloader


class MonoDS(Dataset):
    """A monochromatic dataset class"""

    def __init__(self, src_dir: Path, transform=Compose([ToTensor(), CenterCrop(256)])):
        """
        Parameters:
            base_dir: the base directory for all monochromatic datasets
            wl: the wavelength of the monochromatic channel of interest
            phase: the phase of network for which the dataset should serve
            transform:
        """
        self.src_dir = src_dir
        self._transform = transform

    def __len__(self):
        return len(list(self.src_dir.glob("*.npz")))

    def __getitem__(self, idx):
        image_path = self.src_dir / f"{idx}.npz"
        frame = np.load(image_path)
        fpa_temperature = frame["fpa"].item() / 100
        img = frame["image"].astype(float)
        image_trans = self._transform(img)

        return {"img": image_trans.type(torch.float32), "fpa": fpa_temperature}

    def rec_item(self, img: np.ndarray):
        """A method for reconstructing an image based on the networks output"""
        return img * (self._stats["max"] - self._stats["min"]) + self._stats["min"]


class Tau2AerialDS(Dataset):
    """A dataset class serving the colorization network"""

    def __init__(
        self,
        data_dir: Path,
        mono: LwirChannel = LwirChannel.nm9000,
        is_train: bool = True,
        transform=Compose([ToTensor(), CenterCrop(256)]),
    ):
        self._transform = transform
        self.is_train = is_train
        self.phase = NetPhase.train if is_train else NetPhase.test
        self.pan = MonoDS(
            data_dir / str(LwirChannel.pan) / self.phase.name, transform=transform
        )
        self.mono = MonoDS(data_dir / str(mono) / self.phase.name, transform=transform)
        self.rand_idx = None

    def __len__(self):
        return max(len(self.pan), len(self.mono))

    def __getitem__(self, idx):
        pan_idx = idx
        if self.phase == NetPhase.train:
            if self.rand_idx is None:
                self.rand_idx = self.get_rand_index(
                    len(self.pan)
                )  # TODO: add this code to original branch as well
            try:
                mono_idx = next(self.rand_idx)
            except StopIteration:
                self.rand_idx = self.get_rand_index(len(self.pan))
                mono_idx = next(self.rand_idx)
        else:
            mono_idx = idx

        return {"pan": self.pan[pan_idx], "mono": self.mono[mono_idx]}

    @staticmethod
    def get_rand_index(max_idx):
        """a random index generature invoked during the training loop to avoid overfitting due to always pairing the same panchromatic with the same monochromatic image"""
        rand_order = np.random.choice(range(max_idx), size=max_idx, replace=False)
        for idx in rand_order:
            yield idx

    def rec_item(self, img: np.ndarray, domain):
        """A method for reconstructing an image based on the networks output"""
        return self.__getattribute__(domain.name).rec_item(img)


def main():
    ...


if __name__ == "__main__":
    main()
