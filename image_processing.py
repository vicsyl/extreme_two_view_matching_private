import torch
import torch.nn.functional as F
import math

"""
DISCLAIMER: most of these functions were implemented by me (Vaclav Vavra)
during the MPV course in the Spring semester of 2020, mostly with the help
of the provided template.
"""
def get_gausskernel_size(sigma, force_odd = True):
    ksize = 2 * math.ceil(sigma * 3.0) + 1
    if ksize % 2 == 0 and force_odd:
        ksize +=1
    return int(ksize)


def gaussian_filter2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r"""
    DISCLAIMER: this is a function implemented by me (Vaclav Vavra)
    during the MPV course in spring semester 2020 with the help of the provided
    template.

    Function that blurs a tensor using a Gaussian filter.

    Arguments:
        sigma (Tuple[float, float]): the standard deviation of the kernel.

    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    """
    ksize = get_gausskernel_size(sigma)
    kernel_inp = torch.linspace(-float(ksize // 2), float(ksize // 2), ksize)
    kernel1d = gaussian1d(kernel_inp, sigma).reshape(1, -1)
    outx = filter2d(x, kernel1d)
    out = filter2d(outx, kernel1d.t())
    return out


def gaussian1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    '''
    DISCLAIMER: this is a function implemented by me (Vaclav Vavra)
    during the MPV course in spring semester 2020 with the help of the provided
    template.

    Function that computes values of a (1D) Gaussian with zero mean and variance sigma^2
    '''
    coef = 1./ (math.sqrt(2.0*math.pi)*sigma)
    out = coef*torch.exp(-(x**2)/(2.0*sigma**2))
    return out


def spatial_gradient_first_order(x: torch.Tensor, mask=torch.tensor([[0.5, 0, -0.5]]).float(), smoothed: bool = False, sigma: float = 1.0) -> torch.Tensor:
    r"""
    DISCLAIMER: this is a function implemented by me (Vaclav Vavra)
    during the MPV course in spring semester 2020 with the help of the provided
    template.

    Computes the first order image derivative in both x and y directions using Gaussian derivative

    Return:
        torch.Tensor: spatial gradients

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    """
    #b, c, h, w = x.shape
    if smoothed:
        filtered_input = gaussian_filter2d(x, sigma)
    else:
        filtered_input = x
    outx = filter2d(filtered_input, mask)
    outy = filter2d(filtered_input, mask.t())
    return outx, outy


def filter2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    DISCLAIMER: this is a function implemented by me (Vaclav Vavra)
    during the MPV course in spring semester 2020 with the help of the provided
    template.

    Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(kH, kW)`.
    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    assert len(kernel.size()) == 2
    assert len(x.size()) == 4
    b, c, h, w = x.shape
    height, width = kernel.size()
    tmp_kernel = kernel[None,None,...].to(x.device).to(x.dtype)
    padding_shape =  [width // 2, width // 2, height // 2, height // 2]
    input_pad: torch.Tensor = F.pad(x, padding_shape, mode='replicate')
    out = F.conv2d(input_pad,
                   tmp_kernel.expand(c, -1, -1, -1),
                   groups=c,
                   padding=0,
                   stride=1)
    return out

