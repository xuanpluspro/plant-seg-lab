# MPCNet MPM source

Upstream: https://github.com/Wangtao-Bao/MPCNet
Pinned commit: 5024ea51e8abdf7bc40e36489041143c468b1fdf
Source file: MPCNet.py

model/mpm.py retains the original autopad, activation, CBR, h_sigmoid,
h_swish and CoordAttiton helpers. MPM supports original 7753 behavior and
the project-specific default 5531 variant.
Only imports needed by those definitions are retained; thop is not required.
Attribution remains with the upstream contributors; this extraction is not
claimed as a new module or a reproduction of the full detection network.
No upstream LICENSE file was present at the pinned revision; this file does
not assign the project's Apache license to the extracted upstream code.

Implementation details retained: standard branch convolutions with BN/ReLU;
one shared largest-kernel convolution (including BN) called for two channel groups; concat with
the original input; CBR fusion including its trainable 7x7 depthwise activation;
coordinate attention with h-swish and max(8,C//32) bottleneck. No added QKV
attention, no external residual added to the original MPM.

5531 adaptation: channel groups use 5,5,3,1 in that order. Existing Conv_7/Conv_5/Conv_3 attribute names are retained for serialization, but in 5531 they hold 5x5/3x3/1x1 kernels respectively. The CBR custom 7x7 depthwise activation is unchanged. Original 7753 retains its upstream forward call order.
