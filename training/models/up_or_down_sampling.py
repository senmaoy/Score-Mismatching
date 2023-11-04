"""Layers used for up-sampling or down-sampling images.

Many functions are ported from https://github.com/NVlabs/stylegan2.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from op import upfirdn2d
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d as upfirdn2d_o
from torch_utils.ops import bias_act
from torch_utils.ops import fma

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------

class Conv2dLayer(torch.nn.Module):
  def __init__(self,
                 in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
        ):
    super().__init__()
    self.activation = activation
    self.up = up
    self.down = down
    self.conv_clamp = conv_clamp
    self.register_buffer('resample_filter', upfirdn2d_o.setup_filter(resample_filter))
    self.padding = kernel_size // 2
    self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
    self.act_gain = bias_act.activation_funcs[activation].def_gain

    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
    bias = torch.zeros([out_channels]) if bias else None
    if trainable:
      self.weight = torch.nn.Parameter(weight)
      self.bias = torch.nn.Parameter(bias) if bias is not None else None
    else:
      self.register_buffer('weight', weight)
      if bias is not None:
        self.register_buffer('bias', bias)
      else:
        self.bias = None

  def forward(self, x, gain=1):
    w = self.weight * self.weight_gain
    b = self.bias.to(x.dtype) if self.bias is not None else None
    flip_weight = (self.up == 1) # slightly faster
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

    act_gain = self.act_gain * gain
    act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
    x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
    return x


# Function ported from StyleGAN2
def get_weight(module,
               shape,
               weight_var='weight',
               kernel_init=None):
  """Get/create weight tensor for a convolution or fully-connected layer."""

  return module.param(weight_var, kernel_init, shape)


class Conv2d(nn.Module):
  """Conv2d layer with optimal upsampling and downsampling (StyleGAN2)."""

  def __init__(self, in_ch, out_ch, kernel, up=False, down=False,
               resample_kernel=(1, 3, 3, 1),
               use_bias=True,
               kernel_init=None):
    super().__init__()
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
    if kernel_init is not None:
      self.weight.data = kernel_init(self.weight.data.shape)
    if use_bias:
      self.bias = nn.Parameter(torch.zeros(out_ch))

    self.up = 2 if up else 1
    self.down = 2 if down else 1
    self.resample_kernel = resample_kernel
    self.kernel = kernel
    self.use_bias = use_bias
    self.conv2d = Conv2dLayer(in_ch, out_ch,up=self.up, down = self.down, kernel_size=kernel, bias=False)
    #self.conv2d = Conv2dLayer(in_ch, out_ch, kernel_size=kernel, bias=False)
  def forward(self, x):
    #if self.up:
      #x = upsample_conv_2d(x, self.weight, k=self.resample_kernel)
    #elif self.down:
      #x = conv_downsample_2d(x, self.weight, k=self.resample_kernel)
    #else:
      #x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)
    x = self.conv2d(x)

    if self.use_bias:
      x = x + self.bias.reshape(1, -1, 1, 1)

    return x


def naive_upsample_2d(x, factor=2):
  _N, C, H, W = x.shape
  x = torch.reshape(x, (-1, C, H, 1, W, 1))
  x = x.repeat(1, 1, 1, factor, 1, factor)
  return torch.reshape(x, (-1, C, H * factor, W * factor))


def naive_downsample_2d(x, factor=2):
  _N, C, H, W = x.shape
  x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
  return torch.mean(x, dim=(3, 5))


def upsample_conv_2d(x, w, k=None, factor=2, gain=1):
  """Fused `upsample_2d()` followed by `tf.nn.conv2d()`.

     Padding is performed only once at the beginning, not between the
     operations.
     The fused op is considerably more efficient than performing the same
     calculation
     using standard TensorFlow ops. It supports gradients of arbitrary order.
     Args:
       x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
         C]`.
       w:            Weight tensor of the shape `[filterH, filterW, inChannels,
         outChannels]`. Grouped convolution can be performed by `inChannels =
         x.shape[0] // numGroups`.
       k:            FIR filter of the shape `[firH, firW]` or `[firN]`
         (separable). The default is `[1] * factor`, which corresponds to
         nearest-neighbor upsampling.
       factor:       Integer upsampling factor (default: 2).
       gain:         Scaling factor for signal magnitude (default: 1.0).

     Returns:
       Tensor of the shape `[N, C, H * factor, W * factor]` or
       `[N, H * factor, W * factor, C]`, and same datatype as `x`.
  """

  assert isinstance(factor, int) and factor >= 1

  # Check weight shape.
  assert len(w.shape) == 4
  convH = w.shape[2]
  convW = w.shape[3]
  inC = w.shape[1]
  outC = w.shape[0]

  assert convW == convH

  # Setup filter kernel.
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * (gain * (factor ** 2))
  p = (k.shape[0] - factor) - (convW - 1)

  stride = (factor, factor)

  # Determine data dimensions.
  stride = [1, 1, factor, factor]
  output_shape = ((_shape(x, 2) - 1) * factor + convH, (_shape(x, 3) - 1) * factor + convW)
  output_padding = (output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convH,
                    output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convW)
  assert output_padding[0] >= 0 and output_padding[1] >= 0
  num_groups = _shape(x, 1) // inC

  # Transpose weights.
  w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
  w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4)
  w = torch.reshape(w, (num_groups * inC, -1, convH, convW))

  x = F.conv_transpose2d(x, w, stride=stride, output_padding=output_padding, padding=0)
  ## Original TF code.
  # x = tf.nn.conv2d_transpose(
  #     x,
  #     w,
  #     output_shape=output_shape,
  #     strides=stride,
  #     padding='VALID',
  #     data_format=data_format)
  ## JAX equivalent

  return upfirdn2d(x, torch.tensor(k, device=x.device),
                   pad=((p + 1) // 2 + factor - 1, p // 2 + 1))


def conv_downsample_2d(x, w, k=None, factor=2, gain=1):
  """Fused `tf.nn.conv2d()` followed by `downsample_2d()`.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same
    calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        w:            Weight tensor of the shape `[filterH, filterW, inChannels,
          outChannels]`. Grouped convolution can be performed by `inChannels =
          x.shape[0] // numGroups`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]` or
        `[N, H // factor, W // factor, C]`, and same datatype as `x`.
  """

  assert isinstance(factor, int) and factor >= 1
  _outC, _inC, convH, convW = w.shape
  assert convW == convH
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * gain
  p = (k.shape[0] - factor) + (convW - 1)
  s = [factor, factor]
  x = upfirdn2d(x, torch.tensor(k, device=x.device),
                pad=((p + 1) // 2, p // 2))
  
  return F.conv2d(x, w, stride=s, padding=0)


def _setup_kernel(k):
  k = np.asarray(k, dtype=np.float32)
  if k.ndim == 1:
    k = np.outer(k, k)
  k /= np.sum(k)
  assert k.ndim == 2
  assert k.shape[0] == k.shape[1]
  return k


def _shape(x, dim):
  return x.shape[dim]


def upsample_2d(x, k=None, factor=2, gain=1):
  r"""Upsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and upsamples each image with the given filter. The filter is normalized so
    that
    if the input pixels are constant, they will be scaled by the specified
    `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded
    with
    zeros so that its shape is a multiple of the upsampling factor.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          nearest-neighbor upsampling.
        factor:       Integer upsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]`
  """
  assert isinstance(factor, int) and factor >= 1
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * (gain * (factor ** 2))
  p = k.shape[0] - factor
  return upfirdn2d(x, torch.tensor(k, device=x.device),
                   up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))


def downsample_2d(x, k=None, factor=2, gain=1):
  r"""Downsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and downsamples each image with the given filter. The filter is normalized
    so that
    if the input pixels are constant, they will be scaled by the specified
    `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded
    with
    zeros so that its shape is a multiple of the downsampling factor.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]`
  """

  assert isinstance(factor, int) and factor >= 1
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * gain
  p = k.shape[0] - factor
  k =  torch.tensor(k).cuda()
  
  return upfirdn2d(x, k,
                   down=factor, pad=((p + 1) // 2, p // 2))
