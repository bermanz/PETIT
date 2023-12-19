import datetime

# from configs import AERIAL_RAW_PATH
from pathlib import Path
from shutil import copyfile, rmtree

import numpy as np
import pandas as pd
from p_tqdm import t_map
from tqdm import tqdm

AERIAL_RAW_PATH = Path(r"E:\Datasets\Aerial\flights_data\27_12_21")


def get_img_from_frame(frame_path: Path):
    frame = np.load(frame_path)
    return frame["image"]


def part_dataset(
    target_dir: Path, src_dir: Path, n_train: int, n_val: int, n_test: int
):
    """Create a random partition of the dataset to orthogonal train, validation and test phases

    Parameters:
        n_cv(int): number of cross-validation cycles
    """
    n_tot = n_train + n_val + n_test
    wl = target_dir.stem

    # remove stats of previously partitioned datasets:
    stats_file = target_dir / "stats.json"
    if stats_file.is_file():
        stats_file.unlink()

    # get a random permutation of the source-raw images and use a subset of it to reate the dataset
    frames_dir = src_dir / wl
    partition_log = pd.DataFrame(
        index=range(n_tot), columns=["phase_idx", "frame_num", "phase"]
    )

    # pick a random subset of the dataset for division between the different phase-subsets:
    frame_ids = [frame.stem for frame in frames_dir.glob("*.npz")]
    n_tot = n_train + n_val + n_test
    idx_permute = np.random.choice(frame_ids, size=n_tot, replace=False)

    # copy the source images to the corresponding directory according to the random partition:
    for i, frame_idx in enumerate(
        tqdm(idx_permute, desc=f"parsing meta-data for {wl}"), start=0
    ):
        if i < n_train:
            phase = "train"
            base_idx = 0
        elif i < n_train + n_val:
            phase = "val"
            base_idx = n_train
        else:
            phase = "test"
            base_idx = n_train + n_val
        phase_idx = i - base_idx
        partition_log.loc[i, "frame_num"] = frame_idx
        partition_log.loc[i, "phase"] = phase
        partition_log.loc[i, "phase_idx"] = phase_idx
        frame = np.load(frames_dir / (str(frame_idx) + ".npz"))
        for field in frame.files:
            if field == "image":
                pass  # images are saved by the copy_frames_for_dataset method
            else:
                partition_log.loc[i, field] = frame[field]

    # document the partition in a csv:
    partition_log = partition_log.astype(
        {col: int for col in ["frame_num", "phase_idx", "fpa", "housing"]}
    )
    partition_log.to_csv(target_dir / f"partition_{str(datetime.date.today())}.csv")


def copy_frames_for_dataset(src_dir: Path, target_dir: Path, partition_log):
    # Make directories for the dataset in case these aren't yet available:

    for subdir in ["train", "val", "test"]:
        phase_dir = target_dir / subdir
        if phase_dir.is_dir():
            for path in phase_dir.glob("**/*"):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    rmtree(path)
        phase_dir.mkdir(parents=True, exist_ok=True)

    if (target_dir / "stats.json").is_file():
        (target_dir / "stats.json").unlink()
    wl = target_dir.stem
    src_dirs = [
        src_dir / wl / f"{frame_num}.npz"
        for frame_num in partition_log.frame_num.values
    ]
    target_dirs = [
        target_dir / phase / f"{frame_num}.npz"
        for frame_num, phase in zip(
            partition_log.phase_idx.values, partition_log.phase.values
        )
    ]

    t_map(copyfile, src_dirs, target_dirs, desc=f"copying frames of {wl}")


def gen_dataset(src_dir, n_train: int = 1000, n_val: int = 100, n_test: int = 300):
    """generate the datasets required for the colorization training, validation and testing"""
    target_base_dir = Path.cwd().parent / "data"
    wavelengths = [dir.stem for dir in src_dir.glob("*") if dir.is_dir()]

    target_dirs = [target_base_dir / wl for wl in wavelengths]
    for target_dir in target_dirs:
        target_dir.mkdir(parents=True, exist_ok=True)
        part_dataset(
            target_dir=target_dir,
            src_dir=src_dir,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
        )
        copy_frames_for_dataset(
            src_dir, target_dir, partition_log=get_latest_partition(target_dir)
        )


def get_latest_partition(base_dir: Path):
    partitions = list(base_dir.glob("*partition*.csv"))
    partition_dates = []
    for partition in partitions:
        partition_date_str = partition.stem.split("_")[-1]
        date = [int(num) for num in partition_date_str.split("-")]
        partition_dates.append(datetime.date(*date))

    latest_date = sorted(range(len(partition_dates)), key=lambda k: partition_dates[k])[
        -1
    ]
    return pd.read_csv(partitions[latest_date], index_col=0)


def prep_dataset(
    base_dir: Path = Path(AERIAL_RAW_PATH / "flights_data" / "30_11_21"),
) -> None:
    """A function that combines all the necessary steps for creating the parsing the flight's raw data into valid frames for training.

    As a preliminary step, it is required to download the flight raw-data pkl files from the drive, and unzip them into a clean folder, who's path should be the input to the function.

    It is preferable to create a new directory with the flight's date, as in the default argument provided in the function's decleration"""
    # rafactor_flight_data(base_dir)
    # parse_frames(base_dir)
    # gen_frames_classifier(base_dir)
    # get_land_frames(base_dir)
    gen_dataset(base_dir / "land_frames", n_train=1000, n_val=100, n_test=10_000)


def load_measurements(measurements_path: Path):
    measurements = t_map(
        load_measurement,
        list(measurements_path.glob("*.npz")),
        desc="loading measurements",
    )
    all_dict = {}
    for meas in measurements:
        for key, val in meas.items():
            try:
                all_dict[key] = np.append(all_dict[key], val, axis=0)
            except KeyError:
                all_dict[key] = val
    return all_dict


def load_measurement(meas_file, is_center_crop: bool = True):
    # t_bb = int(meas_file.stem.split("_")[-1])
    raw_meas = np.load(meas_file)
    frames = center_crop(raw_meas["frames"]) if is_center_crop else raw_meas["frames"]
    t_fpa = raw_meas["fpa"] / 100
    t_bb = raw_meas["blackbody"] / 100
    t_bb_vec = np.full_like(t_fpa, t_bb)

    return {"frames": frames, "fpa": t_fpa, "bb": t_bb_vec}


def center_crop(frames):
    frames_spat = frames.shape[-2:]
    spat_diff = frames_spat[1] - frames_spat[0]
    if spat_diff > 0:
        cropped_frames = frames[..., spat_diff // 2 : -spat_diff // 2]
    elif spat_diff < 0:
        cropped_frames = frames[..., spat_diff // 2 : -spat_diff // 2, :]
    else:
        cropped_frames = frames
    return cropped_frames


if __name__ == "__main__":
    # gen_dataset(Path(AERIAL_RAW_PATH / "flights_data" / "27_12_21"/"land_frames"), n_test=10_000)
    # gui = SeaLandSelectGui(Path(r"E:\Datasets\Aerial\flights_data\27_12_21\valid_frames\pan"))
    # gui = SeaLandSelectGui(Path(r"E:\Datasets\Aerial\flights_data\27_12_21\all_frames\9000nm"))
    prep_dataset()
