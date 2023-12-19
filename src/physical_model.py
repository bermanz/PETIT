from functools import partial
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from p_tqdm import t_map
from pre_processing import load_measurements


def c2k(celsius):
    return celsius + 273.15


def k2c(kelvin):
    return kelvin - 273.15


class ThermalRegress:
    """The Multi-Linear-Regressor is designed to conveniently perform parallel multi-channel linear regression between big matrices for the colorization task purpose, either using the multiprocessing toolbox or in a numpy vectorized form."""

    def __init__(self, dtype: Union[np.ndarray, torch.Tensor] = np.ndarray):
        # the physical independent variable (used for plotting)
        # flag for using parallel computing when fitting and predicting
        self.coefficients = None
        self.dtype = dtype

    def _get_features_mat(self, t_fpa, t_bb):
        package = np if isinstance(t_bb, np.ndarray) else torch
        fpa = c2k(t_fpa.flatten())
        bb = c2k(t_bb.flatten()) ** 4
        fpa_features = package.stack([fpa**2, fpa, package.ones_like(t_fpa.flatten())])
        bb_features = bb * fpa_features
        features_mat = package.stack((*bb_features, *fpa_features)).T
        return features_mat

    def get_train_features(self, x: dict):
        """Converts a measurements dictionary into a feature matrix and target vectors"""

        # load data:
        all_frames, t_fpa, t_bb = x["frames"], x["fpa"], x["bb"]

        # shape features and targets:
        features = self._get_features_mat(t_fpa, t_bb)
        target = all_frames.reshape(all_frames.shape[0], -1)
        return features, target, t_fpa, t_bb

    def plot_data_scatter(self, x, pix_idx, label="samples", id=None):
        """Plots a scatter-plot of the input data at the provided pixel"""
        _, radiance, t_fpa, t_bb = self.get_train_features(x)

        if not plt.fignum_exists(id):
            fig = plt.figure(id, figsize=[3.375, 3.375])
            ax = fig.add_subplot(projection="3d")
        else:
            fig = plt.gcf()
            ax = plt.gca()
        # for visualization facilitation - take only one in every 100 samples:
        t_bb = t_bb[::100]
        t_fpa = t_fpa[::100]
        radiance = radiance[::100]
        ax.scatter(t_bb, t_fpa, radiance[:, pix_idx], label=label, alpha=0.5)
        ax.set_xlabel("$T_\mathit{obj}[C]$")
        ax.set_ylabel("$T_\mathit{int}[C]$")
        ax.set_zlabel("Radiometric Intensity")

        return fig, ax

    def fit(self, x: dict, rcond=-1, debug: bool = False):
        """performs a pixel-wise polynomial regression for grey_levels vs an independent variable

        Parameters:
            x: a dictionary containing the calibration measurements (frames and temperatures)
            debug: flag for plotting the regression maps.
        """

        A, b, _, _ = self.get_train_features(x)
        func = partial(np.linalg.lstsq, A, rcond=rcond)
        res = t_map(func, b.T, desc="getting coefficients")
        coeffs_list = [tup[0] for tup in res]
        regress_coeffs = np.asarray(coeffs_list).T
        self.coefficients = regress_coeffs.reshape(
            regress_coeffs.shape[0], *x["frames"].shape[1:]
        )

        if debug:
            # choose random pixel:
            rand_pix_idx_tup = np.random.randint(
                [0, 0], self.coefficients.shape[1:], size=2
            )
            rand_pix_idx = np.ravel_multi_index(
                rand_pix_idx_tup, dims=self.coefficients.shape[1:]
            )

            # scatter the data:
            fig, ax = self.plot_data_scatter(x, rand_pix_idx)

            # evaluate the modeled plane over the scatter's grid:
            ax_lims = ax.xy_viewLim
            x_grid, y_grid = np.meshgrid(
                np.linspace(ax_lims.xmin, ax_lims.xmax),
                np.linspace(ax_lims.ymin, ax_lims.ymax),
            )
            fitted_plane_z = self.predict(x_query=x_grid, t_fpa=y_grid)[
                :, rand_pix_idx_tup[0], rand_pix_idx_tup[1]
            ]
            surf = ax.plot_surface(
                x_grid,
                y_grid,
                fitted_plane_z.reshape(len(x_grid), -1),
                alpha=0.5,
                color="orange",
                label="fitted surface",
            )

            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d

            # lables and final visualizations:
            ax.legend()
            # plt.rcParams.update({'font.size': 22})
            # ax.set_title("Fitted Model ($AT_{BB}^4 + BT_{FPA}^2 + CT_{FPA} + D$)")

    @staticmethod
    def _fpa_like(array, t_fpa):
        reps = len(array.flatten()) // len(t_fpa.flatten())
        if isinstance(t_fpa, np.ndarray):
            t_fpa_rep = t_fpa.repeat(reps)
        elif isinstance(t_fpa, torch.Tensor):
            t_fpa_rep = t_fpa.repeat_interleave(reps)
        return t_fpa_rep.reshape(array.shape)

    def predict(
        self,
        x_query: Union[np.ndarray, torch.Tensor],
        t_fpa: Union[np.ndarray, torch.Tensor],
        direction: str = "temp_to_rad",
    ):
        """Predict the target values by applying the model to the query points.

        Parameters:
            x_query: The query points for the prediction. Should be an ndarray of either radiometric intensities or black-body temperatures.
            t_fpa: The fpa temperature to each of the provided samples [C]
            direction: whether to treat input as temperature and predict radiometry or vice-versa
        Returns:
            results: len(x_query) feature maps, where results[i] corresponds to the predicted features for x_query[i]
        """
        assert self.coefficients is not None
        coeffs_for_pred = self.coefficients.reshape((self.coefficients.shape[0], -1))
        t_fpa = self._fpa_like(x_query, t_fpa)
        features = self._get_features_mat(t_fpa, x_query)
        if direction == "temp_to_rad":
            t_bb = x_query
            if (
                t_bb.shape[-2:] == self.coefficients.shape[-2:]
            ):  # t_bb is an array with image spatial dimensions - need to mulyiply pixel-wise
                features_reshaped = features.reshape(
                    (t_bb.shape[0], -1, features.shape[-1])
                )
                rad_hat = (features_reshaped * coeffs_for_pred.T).sum(axis=-1)
            else:  # t_bb is an array of temperatures - need to take the product with the coefficients matrix
                rad_hat = features @ coeffs_for_pred
            est = rad_hat
        elif (
            direction == "rad_to_temp"
        ):  # inverse model -> T_BB = sqrt_4((rad - B*T_FPA - C) / A)
            radiometry = x_query.reshape(x_query.shape[0], -1)
            n_fpa_feaures = features.shape[1] // 2
            fpa_features = features[:, n_fpa_feaures:]
            features_reshaped = fpa_features.reshape(
                (radiometry.shape[0], -1, fpa_features.shape[-1])
            )
            num = radiometry - (
                features_reshaped * coeffs_for_pred[n_fpa_feaures:].T
            ).sum(axis=-1)
            den = (features_reshaped * coeffs_for_pred[:n_fpa_feaures].T).sum(axis=-1)
            t_bb_hat = k2c((num / den) ** (1 / 4))
            est = t_bb_hat
        else:
            raise Exception("Invalid prediction format!")

        return est.reshape((-1, *self.coefficients.shape[1:]))

    def save(self, target_path: Path):
        np.save(target_path, self.coefficients)

    def load(self, source_path: Path):
        self.coefficients = np.load(source_path)
        if self.dtype == torch.Tensor:
            self.coefficients = torch.from_numpy(self.coefficients)

    def to(self, device):
        assert self.coefficients is not None
        self.coefficients = self.coefficients.to(device)

    def validate(
        self,
        rad: Union[np.ndarray, torch.Tensor],
        t_fpa: Union[np.ndarray, torch.Tensor],
        t_bb: Union[np.ndarray, torch.Tensor],
        direction: str = "temp_to_rad",
        debug: bool = False,
    ):
        if direction == "temp_to_rad":
            x, y = t_bb, rad
        else:
            x, y = rad, t_bb
        pred_func = partial(self.predict, direction=direction)
        batch_sz = 5
        x_batch = x.reshape(x.shape[0] // batch_sz, batch_sz, *x.shape[1:])
        t_fpa_batch = t_fpa.reshape(-1, batch_sz)
        y_hat = np.stack(
            t_map(pred_func, x_batch, t_fpa_batch, desc="predicting validation set")
        ).reshape(*x.shape)
        pred_err = np.moveaxis(y_hat, 0, -1) - y
        rmse = np.sqrt((pred_err**2).mean())
        if debug:
            y_str = "Radiometric" if direction == "temp_to_rad" else "Temperature"
            _, ax = plt.subplots()
            ax.hist(pred_err.flatten())
            ax.set_xlabel("Error")
            ax.set_title(
                f"{y_str} Estimation Error (mean={pred_err.mean():.2f}, std={pred_err.std():.2f}), rmse={rmse:.2f}"
            )

        return pred_err, rmse


class PanToMono:
    def __init__(self, dtype: Union[np.ndarray, torch.Tensor] = np.ndarray) -> None:
        self.pan_model = ThermalRegress(dtype)
        self.mono_model = ThermalRegress(dtype)
        self.dtype = dtype

    def save(self, target_path: Path):
        np.savez(
            target_path,
            pan=self.pan_model.coefficients,
            mono=self.mono_model.coefficients,
        )

    def load(self, source_path: Path):
        model = np.load(source_path)
        self.pan_model.coefficients = (
            model["pan"] if self.dtype == np.ndarray else torch.from_numpy(model["pan"])
        )
        self.mono_model.coefficients = (
            model["mono"]
            if self.dtype == np.ndarray
            else torch.from_numpy(model["mono"])
        )

    def fit(self, x_pan, x_mono):
        print("Fitting Panchromatic Model:")
        self.pan_model.fit(x_pan)
        print("Fitting Monochromatic Model:")
        self.mono_model.fit(x_mono)

    def predict(self, pan_image: np.ndarray, t_fpa_pan: float, t_fpa_mono: float):
        temperature_map = self.pan_model.predict(
            pan_image, t_fpa=t_fpa_pan, direction="rad_to_temp"
        )
        mono = self.mono_model.predict(
            temperature_map, t_fpa=t_fpa_mono, direction="temp_to_rad"
        )
        return mono.squeeze(), temperature_map.squeeze()

    def to(self, device):
        self.pan_model.to(device)
        self.mono_model.to(device)


def comp_meas_profiles():
    """Compare the model estimation error incurred by the measurement profiles"""

    pan_model = ThermalRegress()
    measurements_dict = {
        "bb_cross": "$T_\mathit{int}$ ramp, $T_\mathit{obj}$ constant",
        "sawtooth": "$T_\mathit{int}$ ramp, $T_\mathit{obj}$ triangular",
        "random": "$T_\mathit{int}$ ramp, $T_\mathit{obj}$ random",
    }
    x_val = load_measurements(Path(r"physical_model\measurements\val"))
    val_err = {key: None for key in measurements_dict.keys()}
    _, ax = plt.subplots()
    err_statistics = pd.DataFrame(
        index=measurements_dict.keys(), columns=["MEAN", "STD", "RMSE"]
    )
    for meas_type, label in measurements_dict.items():
        base_path = Path(rf"physical_model\measurements\{meas_type}\pan")
        measurements = load_measurements(base_path)
        pan_model.fit(measurements, debug=False)

        radiometry_pan = x_val["frames"]
        t_fpa_pan = x_val["fpa"]
        bb_temp = x_val["bb"]
        err_vec, rmse = pan_model.validate(
            radiometry_pan, t_fpa_pan, bb_temp, direction="rad_to_temp"
        )

        err_statistics.loc[meas_type, "MEAN"] = err_vec.mean()
        err_statistics.loc[meas_type, "STD"] = err_vec.std()
        err_statistics.loc[meas_type, "RMSE"] = rmse
    return err_statistics


if __name__ == "__main__":
    comp_meas_profiles()
