# Model来源与分割适配

原项目：https://github.com/shuaihao-han/MODA
论文：Han, Shuaihao; Xu, Tingfa; Liu, Peifu; Li, Jianan. MODA: The First Challenging Benchmark for Multispectral Object Detection in Aerial Images. AAAI 2026.

cfb.py、ssaf.py、fem.py从上游tools/my_module/原样复制。随附LICENSE.MODA为上游根目录Apache-2.0许可证；未复制MODA数据或权重。上游源码自己的注释有spectral等词汇，在本RGB版本中这些模块实际处理学习到的通道特征，不代表原始多光谱波段。

| 部分 | 改造 |
|---|---|
| 八波段ResNet-50 | 替换为torchvision RGB ResNet-50 |
| 横向投影 | 用GroupNorm卷积重写 |
| CFB、SSAF、FEM | 上游文件保留，默认以ADPM替换CFB、动态调制替换FEM；SSAF沿用。开关可恢复基线 |
| 框生成的目标图、ActivationLoss | 改为真实植物像素掩膜的BCE监督 |
| 前景引导 | 保留F*(1+sigmoid(aux))残差加权 |
| 旋转框头、额外检测尺度、NMS | 移除 |
| 分割输出 | 新增多尺度解码器、半分辨率RGB细节分支、K类别像素预测（默认二分类） |
| 主损失 | K类交叉熵与前景类别Dice，辅助前景并集BCE权重0.2 |

新适配代码不是作者的官方分割实现；不声称原检测实验复现。所有模型结构与损失在本目录；训练循环只接收接口。新增适配代码按Apache-2.0提供；仓库中原模板文件保留其原有权利归属。发表实验时引用MODA，并明确哪些模块沿用、哪些结构为分割适配。

上游Git blob SHA（2026-09-14读取）：

- cfb.py: 71715072354fbec340ebb566faa7828524ee3d71
- ssaf.py: 7fcbf7e42e8172bf64558ab1bdfedae8ab1feee2
- fem.py: 37dbb5316654f9edf5d3bbcb31f3c658ade7a767

## 候选替代模块（2026-10）

- adpm.py：受MPM多分支感知启发，采用局部/横向/纵向深度卷积和空间分支选择，
  输出保持尺寸，通过原输入残差连接替换最深层CSSP/CFB。不是原版MPM。
- dynamic_fusion.py：受Dino U-Net的FAPM仿射调制启发，改为相邻两层输入，
  调制条件为两层投影和归一化差异；细化保留归一化、激活、可分离卷积和SE，
  捷径改为深层表示。不是原版FAPM或其严格复现；原文未说明的细节是本工程选择。
- 实验使用时需引用MODA、MPCNet与Dino U-Net，检索相关方法后再判断新颖性。
  CPU验证只检查工程和数值契约，不验证植物精度或GPU性能。
