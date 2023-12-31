from functools import partial

import numpy as np
import torch
from dataset import getDataLoader
from inception import InceptionV3
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
from utils.deep import NetPhase


def get_pred(model, batch):
    pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    pred = pred.squeeze(3).squeeze(2).cpu().numpy()
    return pred


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(act):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid(activations):
    """Calculates the FID of two paths"""

    m1, s1 = calculate_activation_statistics(activations["real"])
    m2, s2 = calculate_activation_statistics(activations["fake"])
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def get_inceptionV3(dims=2048, device="cuda"):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    return InceptionV3([block_idx]).to(device)


def get_fid_score(config, model, batch_size=10, inception_dims=2048):
    model.set_phase(NetPhase.test)
    set_loader = getDataLoader(
        config, NetPhase["test"], batch_size=batch_size
    )  # create a Dataset for evaluating the results after each iteration

    mono_for_fid = model.mono_norm._prep_for_eval

    # prepare containers for aggregating all the  visuals:
    inceptionV3 = get_inceptionV3(dims=inception_dims, device="cuda")
    inceptionV3.eval()
    get_inceptionV3_features = partial(get_pred, inceptionV3)

    inception_activations_template = np.empty((len(set_loader.dataset), inception_dims))
    inception_activations = {
        "real": inception_activations_template.copy(),
        "fake": inception_activations_template.copy(),
    }

    # claculate and aggregate visuals:
    with torch.inference_mode():
        for i, data in enumerate(tqdm(set_loader, desc="test checkpoint")):
            model.set_input(data)
            model.forward()

            visuals = model.get_current_visuals()
            for authenticity in inception_activations.keys():
                if config.model == "CycleGan":
                    mono = mono_for_fid(visuals["mono"][authenticity])
                else:
                    mono = mono_for_fid(visuals[f"mono_{authenticity}"])
                inception_activations[authenticity][
                    i * set_loader.batch_size : (i + 1) * set_loader.batch_size
                ] = get_inceptionV3_features(mono)

    return calculate_fid(inception_activations)
