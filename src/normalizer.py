from collections import Counter
import json
import numpy as np
from tqdm import tqdm
from configs import DataNorm, LwirChannel
from configs import TAU2_RAD_RES
from pathlib import Path


class Normalizer:
    """The normalizer class is responsible for handling all the calculations and action involved in the normalization and de-normalization of the TAU2 aerial dataset"""

    def __init__(
        self,
        data_dir: Path,
        wl: LwirChannel,
        norm_method: DataNorm,
        mean: float = 0.0,
        std: float = 0.0,
    ) -> None:
        self.data_dir = data_dir
        self.wl = wl
        self.norm_method = norm_method
        stats = self._get_set_stats()
        self._data_min, self._data_max, self._data_mean, self._data_std = (
            stats["min"],
            stats["max"],
            stats["mean"],
            stats["std"],
        )

        if self.norm_method != DataNorm.const:
            if self.norm_method.name == DataNorm.mean_std.name:
                self._data_offset, self._data_scale = self._data_mean, self._data_std
            elif self.norm_method.name == DataNorm.min_max.name:
                self._data_offset, self._data_scale = (
                    self._data_min,
                    self._data_max - self._data_min,
                )

        self.scale = std
        self.offset = mean

    def normalize(self, x):
        # the 4 factor due to common normalization (pytorch pre-trained models)
        if self.norm_method.name != DataNorm.const.name:
            return ((x - self._data_offset) / self._data_scale) * self.scale + self.offset
        else:
            return x / TAU2_RAD_RES

    def denormalize(
        self, x
    ):  # TODO: Validate perfect reconstruction (x == denormalize(normalize(x)))
        if self.norm_method.name != DataNorm.const.name:
            res = (
                (x - self.offset) / self.scale
            ) * self._data_scale + self._data_offset
        elif self.norm_method.name == DataNorm.const.name:
            res = x * TAU2_RAD_RES
        return res.clip(min=0, max=TAU2_RAD_RES)

    def _prep_for_eval(self, x):
        """prepare the image to be evaluated by the FID metric, which expects normalized values between 0 and 1 as input"""
        x_den = self.denormalize(x)
        x_norm = (x_den - self._data_min) / (self._data_max - self._data_min)
        return x_norm.tile((1, 3, 1, 1))

    def _get_set_stats(self, debug=False) -> dict:
        """Calculates and the pseudo-mean and range of the dataset for normalization purposes. Theses stats reflect the statistics of the entire dataset (regardless of the chosen subsets for training, validation and test) in order to avoid overfitting."""
        base_dir = self.data_dir / str(self.wl)
        base_dir.mkdir(exist_ok=True, parents=True)
        stats_path = base_dir / "stats.json"
        if stats_path.is_file() and not debug:
            with open(base_dir / "stats.json", "r") as fp:
                stats = json.load(fp)
            return stats
        else:
            stat_dirs = [
                "train",
                "val",
            ]  # directories that serve as samples for the distribution analysis (excluding test)
            counter = Counter()
            for sub_dir in stat_dirs:
                for image_path in tqdm(
                    list((base_dir / sub_dir).glob("*.npz")),
                    desc=f"Calculating Statistics from {sub_dir} set",
                ):
                    img = np.load(image_path)["image"]
                    counter.update(img.flatten())

            n_pix_tot = sum(counter.values())
            occurances = list(counter.values())
            norm_occurances = np.asarray(occurances) / n_pix_tot
            intensities = np.asarray(list(counter.keys()))
            mean = np.dot(intensities, norm_occurances)

            intens_diff = intensities - mean
            std = np.sqrt(np.dot(intens_diff**2, norm_occurances))

            stats = {
                "min": min(counter.keys()).astype(float),
                "max": max(counter.keys()).astype(float),
                "mean": mean,
                "std": std,
            }

            if debug:
                import matplotlib.pyplot as plt

                plt.figure()
                plt.bar(counter.keys(), counter.values())
                plt.show()
            else:
                with open(stats_path, "w") as fp:
                    json.dump(stats, fp)

            return counter
