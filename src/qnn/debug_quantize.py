from .quantize import *


class QLinear(nn.Linear):
    def __init__(self,
                 bits,
                 lr,
                 in_features,
                 out_features,
                 bias=True,
                 ):
        super().__init__(in_features, out_features, bias)
        self.qweight = DyQuantize(bits, lr,  [out_features, in_features], 0, name="qweight")
        self.qbias = DyQuantize(8, 0.1,  [1, ], name="qbias")
        self.acc = DyQuantize(8, 0.1,  [1, ], name="qbias")
        self.bits = bits

    def forward(self, x):
        weight = self.qweight(self.weight)
        bias = None
        if self.bias is not None:
            bias = self.qbias(self.bias)
        y0 = F.linear(x, weight, bias)
        y = self.acc(y0)
        return y


class QConv2d(nn.Conv2d):
    def __init__(self,
                 bits, lr,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',  # TODO: refine this type
                 device=None,
                 dtype=None,
                 ):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         groups,
                         bias,
                         padding_mode,
                         device,
                         dtype,
                         )
        self.qweight = DyQuantize(bits, lr,  [1, ], name="qweight")
        self.qbias = DyQuantize(8, 0.1,  [1, ], name="qbias")
        self.acc = DyQuantize(8, 0.1,  [1, ], name="qbias")

    def forward(self, x):
        weight = self.qweight(self.weight)
        bias = None
        if self.bias is not None:
            bias = self.qbias(self.bias)
        # print(weight.shape, x.shape)
        y = F.conv2d(x, weight, bias,
                     self.stride,
                     self.padding,
                     self.dilation,
                     self.groups
                     )

        return self.acc(y)


class QAvgPool2d(nn.AvgPool2d):
    def __init__(self,
                 kernel_size, stride=None, padding=0,
                 ceil_mode: bool = False, count_include_pad: bool = True, divisor_override=None
                 ):
        super().__init__(kernel_size,
                         stride,
                         padding,
                         ceil_mode,
                         count_include_pad,
                         divisor_override,
                         )
        self.acc = DyQuantize(8, 0.1, [1, ], name="qbias")

    def forward(self, x):
        y = F.avg_pool2d(x,
                         self.kernel_size,
                         self.stride,
                         self.padding,
                         self.ceil_mode,
                         self.count_include_pad,
                         self.divisor_override,
                         )
        return self.acc(y)


class QBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.q_acc = DyQuantize(8, 0.1,  [1, ], name="q_acc")
        self.q_w = DyQuantize(8, 0.1,  [1, ], name="q_w")
        self.q_b = DyQuantize(8, 0.1,  [1, ], name="q_b")
        self.q_mean = DyQuantize(8, 0.1,  [1, ], name="q_mean")
        self.q_var = DyQuantize(8, 0.1,  [1, ], name="q_var")

    def forward(self, x):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        y = F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.q_mean(self.running_mean) if not self.training or self.track_running_stats else None,
            self.q_var(self.running_var) if not self.training or self.track_running_stats else None,
            self.q_w(self.weight),
            self.q_b(self.bias),
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        return self.q_acc(y)
