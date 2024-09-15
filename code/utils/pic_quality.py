import numpy as np
from tqdm import tqdm
import torch


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
    min_out, max_out = np.min((min_in - mean) / std), np.max((max_in - mean) / std)  # 正则化min_in 和 max_in
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


def ssim(img1, img2):
    miu1 = img1.mean()
    miu2 = img2.mean()
    sigma1 = np.sqrt(((img1 - miu1) ** 2).mean())
    sigma2 = np.sqrt(((img2 - miu2) ** 2).mean())
    sigma12 = ((img1 - miu1) * (img2 - miu2)).mean()
    k1, k2, L = 0.01, 0.03, 1
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * miu1 * miu2 + C1) / (miu1 ** 2 + miu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim


def mse_psnr(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    psnr = 10 * np.log10(1 ** 2 / mse)
    return mse, psnr


def compute_ssim_mse_psnr(dataloader, noise, net, dataset):
    net.eval()
    ssim_, mse, psnr = 0, 0, 0
    n = 0
    for img, cls, _ in tqdm(dataloader):
        per_images = clamp_img(img + noise, dataset)
        img1 = un_normalize(img[0], dataset=dataset).numpy()
        img2 = un_normalize(per_images[0], dataset=dataset).numpy()
        ssim_ += ssim(img1, img2)
        res = mse_psnr(img1, img2)
        mse += res[0]
        psnr += res[1]
        n += 1
    return float(ssim_ / n), float(mse / n), float(psnr / n)


def compute_ssim_mse_psnr_forpatch(dataloader, applied_patch, mask, net, dataset):
    net.eval()
    ssim_, mse, psnr = 0, 0, 0
    n = 0
    for img, cls, _ in tqdm(dataloader):
        per_images = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + \
                     torch.mul(1 - mask.type(torch.FloatTensor), img.type(torch.FloatTensor))
        img1 = un_normalize(img[0], dataset=dataset).numpy()
        img2 = un_normalize(per_images[0], dataset=dataset).numpy()
        ssim_ += ssim(img1, img2)
        res = mse_psnr(img1, img2)
        mse += res[0]
        psnr += res[1]
        n += 1
    return float(ssim_ / n), float(mse / n), float(psnr / n)
