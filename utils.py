import torch


def psnr(x, y, max_i=1.0, eps=0.000001):
    """Peak Signal-to-Noise Ratio (PSNR). The epsilon is used to counteract error when
    mse=0. Max value is 60, assuming default settings.
    """
    mse = torch.mean((x - y)**2)
    return 10 * torch.log10(max_i/(mse+eps))


def ssim_rgb(x, y, L=1.0, k1=0.01, k2=0.03):
    """Structural Similarity Index Measure (SSIM) for RGB images.
    """
    rgb = len(x.shape) == 3
    if rgb:
        dim = 1
        x = torch.flatten(x, 1)
        y = torch.flatten(y, 1)

    else:
        dim = 0
        x = torch.flatten(x)
        y = torch.flatten(y)

    mu_x = torch.mean(x, dim=dim)
    mu_y = torch.mean(y, dim=dim)
    var_x = torch.var(x, dim=dim)
    var_y = torch.var(y, dim=dim)
    if rgb:
        cov_xy = torch.mean((x-mu_x.unsqueeze(1))*(y-mu_y.unsqueeze(1)),
                            dim=dim)
    else:
        cov_xy = torch.mean((x-mu_x)*(y-mu_y), dim=dim)

    c1, c2 = (k1*L)**2, (k2*L)**2
    out = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) /\
          ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    if rgb:
        out = torch.mean(out)
    return out

def denormalize(imgs):
    imgs = imgs\
        .mul(torch.tensor([0.2738, 0.2607, 0.2856]).view(1,3,1,1))\
        .add(torch.tensor([0.4439, 0.4517, 0.4054]).view(1,3,1,1))
    return torch.clamp(imgs, 0.0, 1.0)

# Both PSNR and SSIM have been used in the NTIRE competition for super-resolution (e.g. https://data.vision.ee.ethz.ch/cvl/ntire18/).
if __name__ == '__main__':
    print(psnr(torch.ones(3, 5, 5), torch.ones(3, 5, 5)))
    print(ssim_rgb(torch.zeros(3, 5, 5), torch.ones(3, 5, 5)))
