import torch
import numpy as np

def load_latents(initial, translated, train_test_ratio=0.8, shuffle=False, device=None):

    # Initializaiton code taken from https://github.com/ngushchin/LightSB/blob/main/notebooks/LightSB_alae.ipynb

    latents = np.load("third_party/LightSB/data/latents.npy")
    gender = np.load("third_party/LightSB/data/gender.npy")
    age = np.load("third_party/LightSB/data/age.npy")


    if initial == "MAN":
        xdata = latents[gender == 'male']
    elif initial == "WOMAN":
        xdata = latents[gender == 'female']
    elif initial == "ADULT":
        xdata = latents[age >= 18 and  (age != -1)]
    elif initial == "CHILDREN":
        xdata = latents[(age < 18) and (age != -1)]
    else:
        raise NotImplementedError

    if translated == "MAN":
        ydata = latents[gender == 'male']
    elif translated == "WOMAN":
        ydata = latents[gender == 'female']
    elif translated == "ADULT":
        ydata = latents[(age >= 18) and (age != -1)]
    elif translated == "CHILDREN":
        ydata = latents[(age <  18) and (age != -1)]
    else:
        raise NotImplementedError

    if shuffle:
        xdata = xdata[torch.randperm(len(xdata)), :]
        ydata = ydata[torch.randperm(len(ydata)), :]

    if device is not None:
        xdata = torch.tensor(xdata, dtype=torch.float32, device=device)
        ydata = torch.tensor(ydata, dtype=torch.float32, device=device)

    i1 = int(train_test_ratio * len(xdata))
    i2 = int(train_test_ratio * len(ydata))

    return xdata[:i1, :], xdata[i1:, :], ydata[:i2, :], ydata[i2:, :]

def sqrtm(matrix: torch.Tensor) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    # vals, vecs = torch.linalg.eigh(matrix)
    vals, vecs = torch.linalg.eig(matrix)
    vals = vals.real
    vecs = vecs.real
    # vals = torch.view_as_complex(vals.contiguous())
    vals_pow = torch.sqrt(vals)
    # vals_pow = torch.view_as_real(vals_pow)[:, 0]
    matrix_pow = vecs @ torch.diag(vals_pow) @ vecs.T
    return matrix_pow


def ALAE_BW(y0, y1):

    # calculate the empirical BW-distance between y0 and y1
    mu0 = y0.mean(dim=0)
    S0 = torch.cov(y0.T)

    mu1 = y1.mean(dim=0)
    S1 = torch.cov(y1.T)

    fid = (mu0 - mu1).norm() ** 2
    # fid = fid + (S0 + S1 - 2 * sqrtm( sqrtm(S0) @ S1 @ sqrtm(S0)) ).diag().sum()
    fid = fid + (S0 + S1 - 2 * sqrtm(S0 @ S1) ).diag().sum()
    return fid

def decode_and_reshape(model, y):

    decoded_img = decode(model, y)
    decoded_img = ((decoded_img * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).permute(0, 2, 3, 1).numpy()
    return decoded_img
