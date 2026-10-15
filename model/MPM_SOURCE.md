# MPCNet MPM source

Upstream: https://github.com/Wangtao-Bao/MPCNet
Pinned commit: 5024ea51e8abdf7bc40e36489041143c468b1fdf
Source file: MPCNet.py

model/mpm.py extracts the original autopad, activation, CBR, h_sigmoid,
h_swish, CoordAttiton and MPM definitions without changing their bodies.
Only imports needed by those definitions are retained; thop is not required.
Attribution remains with the upstream contributors; this extraction is not
claimed as a new module or a reproduction of the full detection network.
No upstream LICENSE file was present at the pinned revision; this file does
not assign the project's Apache license to the extracted upstream code.

Implementation details retained: standard branch convolutions with BN/ReLU;
one shared Conv_7 (including BN) called for two channel groups; concat with
the original input; CBR fusion including its trainable 7x7 depthwise activation;
coordinate attention with h-swish and max(8,C//32) bottleneck. No added QKV
attention, no external residual added to the original MPM.
