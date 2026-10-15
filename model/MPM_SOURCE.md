# MPM-5531 source and adaptation

Upstream: https://github.com/Wangtao-Bao/MPCNet
Pinned commit: 5024ea51e8abdf7bc40e36489041143c468b1fdf
Source file: MPCNet.py
Attribution: Wangtao-Bao/MPCNet contributors.

The autopad, activation, CBR, h_sigmoid, h_swish and CoordAttiton helpers
retain upstream behavior. The MPM is adapted to fixed 5x5, 5x5, 3x3, 1x1
ordinary convolution branches with BN/ReLU. Two 5x5 branches share one
convolution and BN. Branch results concatenate with the original input,
then CBR fuses 2C to C before coordinate attention. The upstream CBR's
trainable 7x7 depthwise activation and coordinate attention's h-swish and
max(8,C//32) bottleneck remain. No QKV attention or extra output residual.

Only required helpers are included; no thop dependency. This is a derived
module, not a new original module or a reproduction of the full network.
No upstream LICENSE file was present at the pinned revision; this document
does not assign this project's Apache license to the upstream code.
