# Plant Seg Lab：RGB 植物二分类语义分割

当前默认模型同时替换 CSSP/CFB 和三个 CAFR/FEM：**ADPM + 跨尺度动态调制**。这是待实验验证的候选结构，不宣称原创性或已提升精度。

基于 [MODA/OSSDet](https://github.com/shuaihao-han/MODA) 的公开特征模块，替换本仓库原 ResNet50-U-Net。网络、模块及损失均在 model/，不依赖 MMRotate、MMDetection、MMCV。这是 OSSDet 衍生的 RGB 分割模型，不是原论文的八波段旋转框检测模型，原论文 mAP 不适用于它。

## 模型

RGB → ResNet-50四级特征 → 通道统一为256 → ADPM深层方向感知 → SSAF自顶向下融合 → 像素监督的植物前景引导 → 跨尺度动态调制自底向上精炼 → 新增多尺度分割解码器与RGB细节分支 → 两通道logits。

- model/ossseg.py：总体网络。默认 ResNet-50，新增分割部分用GroupNorm。
- model/adpm.py：替换CSSP/CFB，1×1降通道→3×3、1×7、7×1深度卷积→逐位置softmax分支选择→投影与残差。
- model/dynamic_fusion.py：替换三个CAFR/FEM，相邻层对齐→联合特征与归一化差异生成空间仿射参数→细化与深层残差。细化保留Conv/GroupNorm/ReLU、深度可分离卷积、SE。
- model/cfb.py、ssaf.py、fem.py：保留官方特征模块，CFB/FEM供消融基线使用，SSAF仍在默认模型中使用。
- model/losses.py：交叉熵 + 前景Dice + 0.2×辅助前景BCE。
- model/README.md：原始模块与改造部分的区别、来源与许可证。

512输入时ADPM在16×16特征上处理；三个动态调制模块输出64×64、32×32、16×16。骨干保留RGB输入，不能产生RGB没有的近红外等信息。三个特征交互模块在AMP中以FP32执行以保护相关性/距离计算；骨干及分割头仍可混合精度运行。


## 同时替换与消融

默认构造 `OSSSeg()` 同时启用两个替代模块，训练无需额外开关。

| 方案 | 训练参数 |
|---|---|
| 两者同时替换（默认） | `--adpm --dynamic-fusion` |
| 原模型基线 | `--no-adpm --no-dynamic-fusion` |
| 仅替换CSSP | `--adpm --no-dynamic-fusion` |
| 仅替换CAFR | `--no-adpm --dynamic-fusion` |

ADPM输入输出均为 `[B,C,h,w]`，先投影到C/4通道，三个分支处理同一压缩特征。
选择器输出 `[B,3,h,w]`，softmax沿分支轴，每个位置三个权重和为1，权重在通道间广播。
动态调制输入 `[B,C,2h,2w]` 与 `[B,C,h,w]`，输出 `[B,C,h,w]`。
三个位置参数独立，依次更新1/8、1/16、1/32尺度。
差异是每个位置沿通道L2归一化后的绝对差，提供线索但不等于语义错误。
调制为 `S*(1+tanh(a))+tanh(b)`，生成器零初始化，使初始缩放为1、偏移为0；
输出为 `T+Refine(M)`。新模块采用GroupNorm，避免小批量BN统计不稳。

结构开关记录在checkpoint中，验证/预测自动恢复。旧checkpoint缺少开关时按旧模型加载；
恢复旧训练需显式传 `--no-adpm --no-dynamic-fusion`。不能把旧完整模型checkpoint直接
恢复为新结构；新实验可使用相同ResNet骨干预训练权重，必须使用新的输出目录。

### AutoDL快速启动

```bash
cd /root/autodl-tmp
git clone https://github.com/xuanpluspro/plant-seg-lab.git
cd plant-seg-lab
pip install -r requirements.txt
python check_data.py --data-path data/VegAnn
python train_medical.py --data-path data/VegAnn --from-scratch --epochs 30 --batch-size 2 --workers 4 --output run/train/adpm_dynamic
```

先以batch-size=2检查真实GPU显存，再提高批大小；尚未验证32GB GPU能承受的最大批大小。
本地ResNet权重可用 `--backbone-weights model/weights/resnet50.pth` 替代 `--from-scratch`。
数据放在data/下且不提交Git；**当前实际读取的是ImageSets/Segmentation，不是Main**。
当前掩膜编码依然要求0/255，不会自动把0/1猜作同一数据。

## 安装

Python 3.10+。先按GPU/CUDA选择配套的torch与torchvision，再安装其余依赖：

~~~bash
pip install -r requirements.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
~~~

CPU测试工作流使用Python3.11、torch2.5.1、torchvision0.20.1；具体运行结果见GitHub Actions。代码支持CPU/CUDA，默认device=auto。单卡起始batch-size=2，显存不足先减为1。改变width会改变实验模型容量，默认256。

## VOC2012 数据

目录如下（文件名是示例）：

| 路径 | 内容 |
|---|---|
| VOCdevkit/VOC2012/JPEGImages/sample001.jpg | RGB图像，允许jpg/jpeg/png等扩展名 |
| VOCdevkit/VOC2012/SegmentationClass/sample001.png | 二值掩膜 |
| VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt | 训练样本ID |
| VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt | 独立验证样本ID |
| VOCdevkit/VOC2012/ImageSets/Segmentation/test.txt | 独立测试样本ID，可选，但需在训练前准备好才能最终测试 |

每行一个同名主干，不带扩展名。图片与掩膜必须都是512×512；程序不悄悄改变尺寸。--data-path接受VOCdevkit或VOC2012路径。不会自行划分；trainval.txt不能当作独立验证集。如果目前只有trainval，先依据真实划分整理为互不重叠的train/val/test。

**磁盘掩膜采用0背景、255植物；读取后转为0背景、1植物。**0/255更易查看，0/1适合交叉熵训练，两者不影响模型精度，只要映射一致。255在这里是植物，不是标准多类VOC中的void。此工程采用VOC目录布局，不采用其21类标签语义。0/1磁盘掩膜需先明确转换，不会自动猜测；彩色可视化也不能作为标签输入。

~~~bash
python check_data.py --data-path VOCdevkit
~~~

检查独立划分、同名配对、尺寸、标签编码。支持索引色PNG的原始索引；真正彩色掩膜报错。验证集不做随机增强；训练同步翻转与90度旋转。图像统一ImageNet mean/std归一化，训练和预测相同。

## ResNet-50预训练

ResNet-50是骨干结构；加载已有权重后才是预训练骨干。原OSSDet配置也使用ResNet-50并指定本地预训练文件。本工程接受本地 **torchvision RGB ResNet-50 ImageNet state_dict**，严格检查键和尺寸，只忽略fc分类器；不自动联网下载。

将权重放在 model/weights/resnet50.pth（该目录及二进制权重未打包），或传任意绝对路径。原U-Net权重、八波段OSSDet检测权重不能直接用。不会在载入后执行旧模板的全模型weights_init。默认冻结骨干BatchNorm运行统计，卷积等参数仍全量训练；--no-freeze-backbone-bn可启用BN统计更新。

## 训练与恢复

使用本地预训练：

~~~bash
python train_medical.py --data-path VOCdevkit --backbone-weights model/weights/resnet50.pth --batch-size 2 --epochs 100 --output run/train/ossseg
~~~

只想检查流程、暂时没有预训练文件，可显式随机初始化：

~~~bash
python train_medical.py --data-path VOCdevkit --from-scratch --batch-size 2 --epochs 2 --output run/train/smoke
~~~

这种短训练不代表最终效果。真实对比应统一预训练来源、数据和训练预算。

~~~bash
python train_medical.py --data-path VOCdevkit --resume run/train/ossseg/last.pth --batch-size 2 --epochs 100 --output run/train/ossseg
~~~

恢复保留原总epochs及其他训练/模型参数，恢复优化器、调度器、AMP与随机状态。不要把--epochs改为剩余轮数。--output须为原运行目录；新实验不能覆盖非空目录。支持--amp/--no-amp，CPU自动禁用AMP。--num-classes沿用旧命令的“前景类别数”含义，固定为1，模型输出2通道。

输出：
- best.pth：验证集**Foreground IoU**最高的checkpoint。
- last.pth：末轮checkpoint，含完整恢复状态。
- config.json、splits.json、history.json：参数、划分和指标记录。
- loss_curve.png、metrics_curve.png：沿用模板绘图。

跨硬件/版本不保证完全一致。划分记录校验ID，不对图像内容做哈希；不要在训练后替换样本。

## 验证、最终测试、预测

~~~bash
python val.py --data-path VOCdevkit --weights run/train/ossseg/best.pth --split val
python val.py --data-path VOCdevkit --weights run/train/ossseg/best.pth --split test
python predict.py --data-path VOCdevkit/VOC2012/JPEGImages --weights run/train/ossseg/best.pth --output run/predict/ossseg
~~~

评估读取checkpoint自带的结构，检查划分与训练时一致。按全数据集混淆矩阵计算前景IoU、背景IoU、mIoU、Dice、Precision、Recall等，不再逐batch简单平均。无定义的指标输出null；mIoU纳入union>0的类。test不用于选择best。

predict输出原尺寸512×512的 *_mask.png（0/255）、*_prob.npy（植物概率）和 *_overlay.png。阈值固定0.5，平局选背景。输出目录须为空，以免覆盖已有预测。当前不包含大图滑窗和实例计数。

## 以后怎样替换模型

~~~python
from model import OSSSeg
model = OSSSeg(num_classes=2)
logits = model(images)                      # [B,2,512,512]
training_output = model(images, return_aux=True)
# {"logits": ..., "aux_logits": [B,1,128,128]}
~~~

训练脚本只依赖此接口及model_config、本地权重加载。后续网络可在model/中适配；若特殊损失/多输入不同，仍需对应修改。此次移除了旧model/MPC.py、resnet_backbone.py、unet_resnet.py、unet_training.py及跟踪的pycache；旧版本留在Git历史。utils下部分旧通用辅助文件保留但不被新训练入口使用。

## 验证

~~~bash
python -m compileall -q model utils train_medical.py val.py predict.py check_data.py tests
python -m unittest discover -s tests -v
~~~

包含默认256通道模型512前向/反向/重载、辅助分支梯度、标签映射、划分泄漏、指标和合成VOC的训练/续训/测试/预测检查。CPU工作流不会下载模型权重、访问用户数据或使用GPU。未在真实植物数据上训练，未验证真实精度及CUDA/AMP表现。原论文的多光谱检测成绩不能作为本改造版的分割成绩。

## 替代模块本地验证记录

2026-10-14，PyTorch 2.5.1 CPU / torchvision 0.20.1：8项unittest通过（23.35秒），
包括默认width=256的512×512前向/反向/重载，三个相邻尺度融合的有限梯度，
四种消融配置及旧格式checkpoint加载，合成VOC训练/中断续训/测试/预测。
默认双替换模型参数36,611,682，原基线33,715,104；保留完整细化路径使总参数增加约8.6%。
没有真实数据精度结果，CUDA/AMP尚未验证。docs/figures下旧结构图描述的是原基线，
不代表当前ADPM+动态调制默认结构。
