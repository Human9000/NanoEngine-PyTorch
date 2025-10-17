import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Any, List, Callable
import operator
from torch.onnx.symbolic_helper import _get_tensor_sizes
from torch.onnx import symbolic_helper
import copy

import numpy as np
import math
import pandas as pd


# --- Base Quantization Functions (unchanged) ---
# @torch.no_grad()
def _fake_q_scale(
        x: torch.Tensor,
        bits_weight: int,
        scale: torch.Tensor,
        name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert bits_weight in [1, 2, 4, 8], "bits_len must be 1, 2, 4, or 8"
    device, dtype = x.device, x.dtype

    if bits_weight == 1:
        q = torch.where(x >= 0,
                        torch.ones_like(x, dtype=torch.float32),
                        -torch.ones_like(x, dtype=torch.float32)) * scale
        # mask = (x.abs() <= scale)
        mask = (x.abs() <= 1)
    else:
        trunc_x_scale = torch.trunc(x * scale)
        q_val = torch.clamp(
            trunc_x_scale,
            - 2 ** (bits_weight - 1),
            2 ** (bits_weight - 1) - 1,
        )
        q = q_val / scale
        mask = (q_val == trunc_x_scale)

    return q.to(dtype=dtype, device=device), mask.detach()
    # return x, mask.detach()


def q1_pack_bits(q_data: np.ndarray) -> np.ndarray:
    """
    Q1 bit packing: 每 8 个 bit 打包成 1 个 uint8
    q_data: 输入 ndarray, 值为 0 或 1
    返回: uint8 packed array
    """
    # 确保是 uint8
    q_data = q_data.astype(np.uint8)

    # padding 到 8 的倍数
    pad_len = (-q_data.size) % 8
    if pad_len:
        q_data = np.pad(q_data, (0, pad_len), constant_values=0)

    # reshape 每行 8 个 bit
    q_data = q_data.reshape(-1, 8)

    # 按位打包
    shifts = np.arange(7, -1, -1)  # bit7 到 bit0
    packed = np.zeros(q_data.shape[0], dtype=np.uint8)
    for i in range(8):
        packed |= q_data[:, i] << shifts[i]
    return packed


def q_packed_bits(
        q_data: np.ndarray,  # 输入数据,已经做过量化缩放的量化数据
        q_bits: int,  # 量化的位数
) -> Tuple[
    np.ndarray,  # 量化后的张量（同 x 的 dtype & device）
    np.ndarray,  # 用于截断梯度的掩码（float，和 x 同 dtype）
]:
    assert q_bits in (1, 2, 4, 8), "只支持 i1, i2, i4, i8"

    q_data = q_data.astype(np.int8)  # 保留符号

    if q_bits == 8:
        return q_data.flatten()

    pack_size = 8 // q_bits
    pad_len = -q_data.shape[-1] % pack_size
    if pad_len != 0:
        pad_size = tuple([(0, 0), ] * (len(q_data.shape) - 1) + [(0, pad_len,)])
        q_data = np.pad(q_data, pad_size, constant_values=0)  # 用 0 填充到 4 的倍数

    q_data = q_data.reshape(-1, pack_size).astype('int8')

    shifts = np.arange(pack_size - 1, -1, -1).astype('int8') * q_bits
    packed = np.zeros(q_data.shape[0], dtype=np.int8)

    # 使用补码表示打包
    for i in range(pack_size):
        mask = (np.ones(1, dtype='int8') << q_bits) - 1
        d = (q_data[:, i] & mask)
        packed |= d << shifts[i]
    return packed.tobytes()


class DyQuantizedWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, bit_len, scale, name) -> torch.Tensor:
        q_tensor, q_mask = _fake_q_scale(tensor, bit_len, scale, name)
        ctx.q_mask = q_mask
        return q_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Any, ...]:
        return grad_output * ctx.q_mask, None, None, None


class ConstQuantizedWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, bit_len, scale: torch.Tensor, name: str) -> torch.Tensor:
        q_tensor, q_mask = _fake_q_scale(tensor, bit_len, scale, name)
        ctx.q_mask = q_mask
        ctx.bit_len = bit_len
        return q_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Any, ...]:

        return grad_output * ctx.q_mask, None, None, None

    @staticmethod
    def symbolic(g, tensor, bit_len, scale, name):
        scale_tensor = symbolic_helper._node_get(scale.node(), 'value')
        scale_k_tensor = torch.log2(scale_tensor).to(torch.int8)
        # scale_k_tensor 封装成一个新的const
        scale_k_tensor = g.op("Constant", value_t=scale_k_tensor.clone().detach())
        q_tensor = g.op("t2e.qmodel::Q", tensor, scale_k_tensor, bit_len_i=bit_len)
        # q_tensor = g.op("t2e.qmodel::Q", tensor, scale_k_tensor=scale_k_tensor, bit_len_i=bit_len)
        # 获取输入 tensor 的 shape 和 dtype
        input_shape = _get_tensor_sizes(tensor)
        input_dtype = tensor.type().dtype()
        # 遍历input_shape, 如果有none，替换成-1
        none_value = -1
        for i in range(len(input_shape) - 1, -1, -1):
            if input_shape[i] is None:
                input_shape[i] = none_value
                none_value -= 1
        # 显式设置输出类型
        q_tensor.setType(tensor.type().with_sizes(input_shape).with_dtype(input_dtype))

        return q_tensor


def dy_quantize(tensor, bit_len, scale, name):
    return DyQuantizedWrapper.apply(tensor, bit_len, scale, name)


def const_quantize(tensor, bit_len, scale, name):
    return ConstQuantizedWrapper.apply(tensor, bit_len, scale, name)


class DyQuantize(nn.Module):
    def __init__(self,
                 bits_len: int = 8,
                 lr: float = 0.1,
                 in_shape: Tuple[int] = None,
                 c_out_dim: int = None,
                 use_channel_scale: bool = False,
                 init_scale: torch.Tensor = None,
                 name="",
                 ):
        super().__init__()
        self.lr = lr
        self.bits_len = bits_len
        self.channel_max_dims = None
        self.name = name
        self.scale_shape = [1, ]
        self.in_shape = in_shape
        self.use_channel_scale = use_channel_scale
        self.c_out_dim = c_out_dim
        channel = 1
        self.upper_bound = 2.0 ** (bits_len - 1)
        if init_scale is None:
            if in_shape is not None:
                for i, v in enumerate(in_shape):
                    if i != c_out_dim:
                        channel *= v
                if c_out_dim is not None:
                    if use_channel_scale:
                        self.scale_shape = [1, ] * len(in_shape)
                        self.scale_shape[c_out_dim] = in_shape[c_out_dim]
                        self.channel_max_dims = [i for i in range(len(in_shape)) if i != c_out_dim]
            else:
                self.in_shape = []

            init_scale = torch.ones(self.scale_shape, requires_grad=False)
            if bits_len == 1:
                self.upper_bound = 1
                init_scale *= channel ** -0.5
            else:
                init_scale *= 2 ** (bits_len - 1)

        self.register_buffer('init_scale', init_scale)
        self.register_buffer('scale', init_scale)
        self.count = 0

    def to_const(self):
        print(self.name, self.scale_shape, self.scale.shape)
        return ConstQuantize(self.bits_len, self.get_exp_clip_log2_scale(), self.name)

    @torch.no_grad()
    def _update_scale(self, x):
        xabs = x.detach().abs()

        if self.use_channel_scale is not False:
            if len(self.channel_max_dims) > 0:
                xabsmax = torch.amax(xabs, dim=self.channel_max_dims, keepdim=True)
            else:
                xabsmax = xabs
        else:
            xabsmax = xabs.max()

        if self.bits_len == 1:
            scale = xabsmax
        else:
            scale = self.upper_bound / xabsmax
        mask = xabsmax < 1e-6
        scale[scale > 2] *= 0.9  # init_scale 过大则提供一个向下移动的方向
        scale[scale < 0.5] /= 0.9  # init_scale 过小则提供一个向上移动的方向
        updated_scale = self.scale * (1 - self.lr) + self.lr * scale  # 用 init_scale 更新
        updated_scale[mask] = self.scale[mask]  # xabsmax 过小则保持不变

        if self.scale.shape != updated_scale.shape:
            print("=====", self.name, self.scale.shape, updated_scale.shape, xabsmax.shape)
        self.scale = updated_scale

    def get_scale_k(self):
        scale_k = torch.log2(self.scale)
        scale_k = torch.floor(scale_k)  # 向下取整
        return scale_k.detach()

    def get_exp_clip_log2_scale(self):
        return (2 ** self.get_scale_k()).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_scale(x)
        q = dy_quantize(x, self.bits_len, self.get_exp_clip_log2_scale(), self.name)

        self.count += 1

        if self.bits_len == 8:
            s0 = self.scale.item()
            s = self.get_exp_clip_log2_scale().item()
            # _fake_q_scale(x, 8, self.acc.get_exp_clip_log2_scale(), )
            if self.count % 100 == 1:
                if "bn" in self.name:
                    print(
                        self.count,
                        self.name,
                        x.abs().max().item(),
                        x.abs().max().item() * s,
                        s0,
                        s,
                    )
        return q

    def extra_repr(self) -> str:
        return f'bits_len={self.bits_len}, ' \
               f'lr={self.lr}, ' \
               f'in_shape={self.in_shape}, ' \
               f'scale_shape={self.scale_shape}, ' \
               f'c_out_dim={self.c_out_dim}, ' \
               f'USE_CS={self.use_channel_scale}, '


class ConstQuantize(nn.Module):
    def __init__(self, bits_len: int, init_scale: torch.Tensor, name=""):
        super().__init__()
        self.bits_len = bits_len
        self.scale = init_scale
        self.register_buffer('scale_k', self.get_scale_k())
        self.name = name

    def get_scale_k(self):
        scale_k = torch.log2(self.scale).round()
        return scale_k

    def get_scale(self):
        scale_k = torch.log2(self.scale).round()
        scale_k = torch.clip(scale_k, 1, self.bits_len - 1).int()
        return 2 ** scale_k

    def to_dy(self, lr):
        return DyQuantize(self.bits_weight, lr, init_scale=self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return const_quantize(x, self.bits_len, self.scale, self.name)

    def extra_repr(self) -> str:
        return f'bits_len={self.bits_len}, shape={list(self.scale.shape)}'
