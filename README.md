# Plant Seg Lab：RGB 植物语义分割

当前默认模型同时替换 CSSP/CFB 和三个 CAFR/FEM：**ADPM + 跨尺度动态调制**。这是待实验验证的候选结构，不宣称原创性或已提升精度。

基于 [MODA/OSSDet](https://github.com/shuaihao-han/MODA) 的公开特征模块，替换本仓库原 ResNet50-U-Net。网络、模块及损失均在 model/，不依赖 MMRotate、MMDetection、MMCV。这是 OSSDet 衍生的 RGB 分割模型，不是原论文的八波段旋转框检测模型，原论文 mAP 不适用于它。

## 模型

RGB → ResNet-50四级特征 → 通道统一为256 → ADPM深层方向感知 → SSAF自顶向下融合 → 像素监督的植物前景引导 → 跨尺度动态调制自底向上精炼 → 新增多尺度分割解码器与RGB细节分支 → K通道logits（默认二分类）。

- model/ossseg.py：总体网络。默认 ResNet-50，新增分割部分用GroupNorm。
- model/adpm.py：替换CSSP/CFB，1×1降通道→3×3、1×7、7×1深度卷积→逐位置softmax分支选择→投影与残差。
- model/dynamic_fusion.py：替换三个CAFR/FEM，相邻层对齐→联合特征与归一化差异生成空间仿射参数→细化与深层残差。细化保留Conv/GroupNorm/ReLU、深度可分离卷积、SE。
- model/cfb.py、ssaf.py、fem.py：保留官方特征模块，CFB/FEM供消融基线使用，SSAF仍在默认模型中使用。
- model/losses.py：K类交叉熵 + 前景类别Dice + 0.2×辅助前景并集BCE。
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
python download_pretrained.py
python train_medical.py --data-path data/VegAnn --epochs 30 --batch-size 32 --workers 4 --output run/train/adpm_dynamic
```

默认batch-size=32、epochs=30；实际显存不足时降低batch-size。
本地ResNet权重可用 `--backbone-weights model/weights/resnet50.pth` 替代 `--from-scratch`。
数据放在data/下且不提交Git；**当前实际读取的是ImageSets/Segmentation，不是Main**。
标签支持0/255、0/1二分类，以及连续类别编号的多分类，参数见下方。

## 安装

Python 3.10+。先按GPU/CUDA选择配套的torch与torchvision，再安装其余依赖：

~~~bash
pip install -r requirements.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
~~~

CPU测试工作流使用Python3.11、torch2.5.1、torchvision0.20.1；具体运行结果见GitHub Actions。代码支持CPU/CUDA，默认device=auto。默认batch-size=32，显存不足可降低。改变width会改变实验模型容量，默认256。

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

**背景为0，保留P模式PNG类别索引，不把显示颜色当作类别。**

| 原始标签 | 训练和check_data参数 |
|---|---|
| 0、255 | `--mask-encoding binary255 --num-classes 1` |
| 0、1 | `--mask-encoding index --num-classes 1` |
| 0、1、2、3、4 | `--mask-encoding index --num-classes 4` |

`--num-classes`是**不含背景**的类别数。默认`auto`在二分类接受0/1或0/255，多分类保留索引，越界报错。不支持ignore标签；255在binary255中表示前景，不是VOC的void。彩色可视化标签需先转类别索引。

~~~bash
python check_data.py --data-path VOCdevkit
~~~

检查独立划分、同名配对、尺寸、标签编码。支持索引色PNG的原始索引；真正彩色掩膜报错。验证集不做随机增强；训练同步翻转与90度旋转。图像统一ImageNet mean/std归一化，训练和预测相同。

## ResNet-50预训练

ResNet-50是骨干结构；加载已有权重后才是预训练骨干。原OSSDet配置也使用ResNet-50并指定本地预训练文件。本工程接受本地 **torchvision RGB ResNet-50 ImageNet state_dict**，严格检查键和尺寸，只忽略fc分类器；不自动联网下载。

默认权重位置为 `pretrained/resnet50.pth`。运行 `python download_pretrained.py` 下载官方 torchvision ResNet-50 ImageNet-1K V2 权重（验证SHA256前缀）；文件不提交Git。也可通过 `--backbone-weights` 指定其他兼容本地权重。训练默认加载此文件，缺少文件会报错，不会自动改为随机初始化。原U-Net权重、八波段OSSDet检测权重不能直接用。不会在载入后执行旧模板的全模型weights_init。默认冻结骨干BatchNorm运行统计，卷积等参数仍全量训练；--no-freeze-backbone-bn可启用BN统计更新。

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

恢复保留原总epochs及其他训练/模型参数，恢复优化器、调度器、AMP与随机状态。不要把--epochs改为剩余轮数。--output须为原运行目录；新实验不能覆盖非空目录。支持--amp/--no-amp，CPU自动禁用AMP。--num-classes沿用旧命令的“前景类别数”含义，二分类填1，五分类填4；模型输出通道包含背景。恢复本次升级前的checkpoint还需加`--mask-encoding binary255`，原有消融开关和训练参数保持一致。

输出：
- best.pth：二分类按验证集**Foreground IoU**、多分类按验证集**Mean IoU**最高保存。
- last.pth：末轮checkpoint，含完整恢复状态。
- config.json、splits.json、history.json、history.csv：参数、划分、逐轮指标、学习率和耗时记录。训练和验证均显示tqdm进度条。
- loss_curve.png、metrics_curve.png：沿用模板绘图。

跨硬件/版本不保证完全一致。划分记录校验ID，不对图像内容做哈希；不要在训练后替换样本。

## 验证、最终测试、预测

~~~bash
python val.py --data-path VOCdevkit --weights run/train/ossseg/best.pth --split val
python val.py --data-path VOCdevkit --weights run/train/ossseg/best.pth --split test
python predict.py --data-path VOCdevkit/VOC2012/JPEGImages --weights run/train/ossseg/best.pth --output run/predict/ossseg
~~~

评估读取checkpoint自带的结构，检查划分与训练时一致。按全数据集混淆矩阵计算前景IoU、背景IoU、mIoU、Dice、Precision、Recall等，不再逐batch简单平均。无定义的指标输出null；mIoU纳入union>0的类。test不用于选择best。

predict输出512×512的 *_mask.png、*_prob.npy和 *_overlay.png。二分类默认0/255与H×W前景概率，可用`--mask-output-encoding index`导出0/1；二分类阈值等效0.5，平局选背景。多分类用argmax，PNG保留0..K-1，概率为K×H×W。输出目录须为空，以免覆盖已有预测。当前不包含大图滑窗和实例计数。

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

## 参数量、计算量与推理速度

参考 xuanpluspro/vegann-seg-benchmark 的统计口径，训练结束后默认加载best.pth，
自动生成运行目录中的 **benchmark.json**。原来的分割指标、best模型选择依据不变。

- parameters / params_m：总参数个数 / 百万参数。
- counted_gflops：fvcore能够计数的操作；一次乘加计为一次操作，与按两次计数的工具不能直接比较。
- gflops：只有没有unsupported_ops时才填写完整统计值，否则为null。
- flops_complete / unsupported_ops / flops_error：计算量统计完整性及遗漏、错误。
- latency_ms / latency_median_ms / latency_p95_ms：平均、中位数和P95单图GPU前向耗时。
- fps：1000/平均毫秒耗时，batch固定为1。
- precision / gpu_name / input_shape / 环境版本：记录可比较的测试条件。

测速为PyTorch eager FP32，关闭TF32和autocast，输入默认1×3×512×512，预热50次，
测量200次，使用CUDA events同步。范围是模型前向返回logits，不包含读取图片、
预处理、CPU/GPU传输、后处理或文件保存；不是TensorRT，也不代表端到端FPS。
训练AMP开关与这项FP32测速相互独立。CPU可计数FLOPs，但GPU FPS和延迟为null。
含自定义/非支持算子时，请明确报告“部分GFLOPs”，不能将counted_gflops当成完整计算量。

拉取后更新新增依赖：

```bash
git pull
pip install -r requirements.txt
```

已有checkpoint也能单独测速，不需要数据集或重新训练：

```bash
python benchmark.py --weights run/train/riceseg_adpm_dynamic/best.pth --device cuda
```

训练可用 `--benchmark-warmup 50 --benchmark-iterations 200` 调整次数；
`--no-benchmark` 可关闭训练末尾的自动统计，之后再单独执行benchmark.py。

## 默认使用预训练骨干（AutoDL）

```bash
cd /root/autodl-tmp/plant-seg-lab && git pull && python download_pretrained.py
python train_medical.py --data-path data/VOC2012 --epochs 30 --batch-size 32 --workers 4 --device cuda --amp --output run/train/riceseg_pretrained_bs32_ep30
```

预训练仅初始化ResNet-50，两个新模块和分割头没有对应预训练权重，需训练。
当前默认冻结骨干BN运行统计，骨干卷积仍训练。若batch=32显存不足，降低批大小。
显式 `--from-scratch` 仍用于公平消融或CPU测试；`--resume` 恢复完整checkpoint，
不要求另外准备预训练文件。输出目录必须新建且为空。

## 训练结束的最终结果（自动生成）

参考vegann-seg-benchmark，训练结束自动加载 **best.pth**，重新评估验证集和训练时
已经登记的独立测试集，再统计性能。测试集不参与选择best。输出目录新增：

| 文件 | 内容 |
|---|---|
| metrics.json | 最佳epoch、验证/测试精度、模型参数和benchmark汇总 |
| results.csv | 验证/测试各一行，可用Excel打开；区域指标0~1，HD95/ASSD单位为像素 |
| class_metrics.csv | 每类IoU、Accuracy、Dice、Precision、Recall和真实像素数 |
| benchmark.json | 参数量、完整性标记、部分/完整GFLOPs及GPU FPS/延迟 |

终端也会打印最终前景IoU、mIoU、Dice、Precision、Recall和PixAcc。
没有test.txt时只生成验证集结果，并明确注明test未提供，不会用验证集冒充测试集。
`--no-benchmark`只关闭性能测速，仍导出最终精度结果。先保存精度结果，再测速，
即使测速失败也保留metrics.json和results.csv，并记录测速错误。

已经训练完成的模型无需重新训练，更新代码后执行：

```bash
python summarize.py --data-path data/VOC2012 --weights run/train/riceseg_pretrained_bs32_ep30/best.pth --device cuda --batch-size 32 --workers 4
```

数据划分必须与checkpoint登记的train/val/test完全相同。输出默认位于权重目录；
`--output run/report/riceseg`可生成到另一目录，不修改权重文件。


## 与参考基准的指标口径

参考[vegann-seg-benchmark](https://github.com/xuanpluspro/vegann-seg-benchmark)，区域指标累计整个数据集混淆矩阵；新增Specificity、HD95、ASSD和Boundary F1。边界指标逐图平均，容差固定2像素；双方前景为空时距离0/F1为1，单方为空时距离为图像对角线/F1为0。模型训练仍采用本项目的CE+Dice+辅助BCE，不直接替换成参考仓库的单通道BCE。

多分类Dice/Precision/Recall是前景类别宏平均，mIoU包含背景；Foreground IoU和边界指标衡量所有非背景类别并集，需同时查看逐类IoU以判断类别混淆。无定义区域分数输出null，宏平均排除未定义项；这一空类别约定与参考仓库的返回1约定不同。metric_units和aggregation记录单位与汇总方法，metric_scale的0-1仅适用于比例指标。

训练结束自动用best.pth评估val/test并输出结果，测试集不参与选模。新checkpoint记录类别数、编码和模块开关，验证/预测/汇总自动复用；旧checkpoint仍可加载，缺少标签配置时按0/255二分类处理。已有模型可用summarize.py重新生成新增指标，无需重训。历史训练指标不会追溯补算。

当前0/1数据集示例：

```bash
python check_data.py --data-path data/VOC201201 --mask-encoding index --num-classes 1
python train_medical.py --data-path data/VOC201201 --mask-encoding index --num-classes 1 --backbone-weights pretrained/resnet50.pth --batch-size 32 --epochs 30 --output run/train/vegann01_pretrained_bs32_ep30
```

保持512×512图像与掩码；本次没有添加隐式缩放。所有对比实验应统一划分、预训练来源、标签语义、尺寸和统计口径。真实CUDA精度与性能需在AutoDL验证。
