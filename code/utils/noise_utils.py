import numpy as np
import torch


def noise_initialization(image_size=(3, 224, 224)):
    noise = np.zeros(image_size).astype('float32')
    return noise


def clamp_noise(noise, dataset):
    if dataset == 'CASIA':
        mean = [0.496, 0.385, 0.324]
        std = [0.284, 0.245, 0.236]
    elif dataset == 'vggfaces2':
        mean, std = [0.596, 0.456, 0.390], [0.263, 0.228, 0.219]
    else:
        raise NotImplementedError('Only CASIA and vggfaces2 are implemented currently.')

    # min 0, max 1
    min_in = np.array([0, 0, 0])
    max_in = np.array([1, 1, 1])
    min_out, max_out = np.min((min_in - mean) / std), np.max((max_in - mean) / std)
    noise = torch.clamp(noise, min=min_out, max=max_out)
    noise = torch.clamp(noise, min=-16 / 255, max=16 / 255)
    return noise


def clamp_img(per_img, dataset):
    if dataset == 'CASIA':
        mean = [0.496, 0.385, 0.324]
        std = [0.284, 0.245, 0.236]
    elif dataset == 'vggfaces2':
        mean, std = [0.596, 0.456, 0.390], [0.263, 0.228, 0.219]
    else:
        raise NotImplementedError('Only CASIA and vggfaces2 are implemented currently.')

    min_in = np.array([0, 0, 0])
    max_in = np.array([1, 1, 1])
    min_out, max_out = np.min((min_in - mean) / std), np.max((max_in - mean) / std)
    per_img = torch.clamp(per_img, min=min_out, max=max_out)
    return per_img


def un_normalize(x, dataset):
    if dataset == 'CASIA':
        mean, std = [0.496, 0.385, 0.324], [0.284, 0.245, 0.236]
    elif dataset == 'vggfaces2':
        mean, std = [0.596, 0.456, 0.390], [0.263, 0.228, 0.219]
    else:
        raise NotImplementedError('Only CASIA and vggfaces2 are implemented currently.')

    with torch.no_grad():
        x[0] = x[0] * std[0] + mean[0]
        x[1] = x[1] * std[1] + mean[1]
        x[2] = x[2].mul(std[2]) + mean[2]
    return x
