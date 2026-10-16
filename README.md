# Plant Seg Lab：RGB 植物语义分割

当前固定使用 **MPM-5531** 替换 CSSP，并使用跨尺度动态调制替换三个 CAFR。支持二分类和多分类。该组合的真实精度需要实验验证，不预先宣称提升。

## 模型结构

RGB → ResNet-50四级特征 → 通道统一 → 最深层MPM-5531 → DRM/CBM自顶向下融合 → 辅助前景引导 → 三个跨尺度动态调制 → 多尺度解码器＋RGB细节分支 → K类logits。

- `model/mpm.py`：通道四分，依次经过5×5、5×5、3×3、1×1普通卷积及BN/ReLU；两个5×5分支共享卷积和BN。四路结果与原输入Concat到2C，再由CBR融合回C，最后经过CAM。CBR保留MPCNet的可学习7×7深度卷积激活；CAM采用h-swish。未加入QKV。
- `model/dynamic_fusion.py`：浅层降采样与深层投影，共同生成仿射调制参数；细化后加深层残差。三个位置参数独立。
- `model/ssaf.py`：SACF中的SFA替换为DRM（深层），SDE替换为CBM（浅层）；保留原来的双线性上采样、双Sigmoid空间融合权重和Conv/BN/ReLU输出投影。
- `model/fem.py`：保留原CAFR/FEM，供`--no-dynamic-fusion`对照使用。
- `model/losses.py`：K类交叉熵＋前景Dice＋0.2×辅助前景并集BCE。

512×512输入时，四层统一后的特征为[B,256,128,128]、[B,256,64,64]、[B,256,32,32]、[B,256,16,16]，MPM位于最后一层。动态调制依次更新1/8、1/16、1/32特征。默认宽度256，`--width`可调整。特征交互路径使用FP32，骨干和分割头支持AMP。

当前不提供旧CSSP实现选择、旧模型加载适配或卷积核切换。固定5531模型采用新的检查点格式；旧完整模型请使用Git历史版本评估。新实验从ResNet骨干预训练权重开始，不覆盖旧结果目录。数据集、权重文件不会随更新删除。

## AutoDL 安装与更新

使用相互匹配的PyTorch和torchvision（PyTorch≥2.4），然后：

```bash
cd /root/autodl-tmp
git clone https://github.com/xuanpluspro/plant-seg-lab.git
cd plant-seg-lab
pip install -r requirements.txt
```

已有仓库只需在没有训练运行时执行：

```bash
cd /root/autodl-tmp/plant-seg-lab
git pull --ff-only origin main
pip install -r requirements.txt
```

## 数据目录与标签

把不同数据集放到`data/`下，不提交Git。`--data-path`指向含JPEGImages、SegmentationClass、ImageSets/Segmentation的VOC根目录。清单为train.txt、val.txt和可选test.txt，每行一个无扩展名ID。划分必须互不重叠，训练前准备好测试清单，不自动重新划分。

图像和掩码都必须512×512，不隐式缩放；训练同步翻转与90度旋转，验证无随机增强。RGB按ImageNet mean/std归一化。

| 掩码像素 | 参数 |
|---|---|
| 0背景、255前景 | `--mask-encoding binary255 --num-classes 1` |
| 0背景、1前景 | `--mask-encoding index --num-classes 1` |
| 0、1、2、3、4 | `--mask-encoding index --num-classes 4` |

`--num-classes`不包含背景。默认auto接受二分类0/1或0/255；多分类保留连续类别ID。P模式PNG保留原索引，彩色RGB标签需先转索引。不支持忽略标签；binary255中的255是前景。

```bash
python check_data.py --data-path data/VOC201201 --mask-encoding index --num-classes 1
```

## 预训练与训练

默认使用`pretrained/resnet50.pth`，缺失时报错，不回退随机初始化。可用`python download_pretrained.py`下载torchvision官方ResNet-50 ImageNet-1K V2权重，或手动放入兼容的RGB骨干state_dict。只初始化骨干，分割模块需要训练。

默认batch32、30轮，AdamW，学习率1e-4，余弦调度，AMP开启。骨干卷积仍更新；`freeze_backbone_bn`只固定骨干BN运行统计，MPM内部BN照常训练。

```bash
python train_medical.py \
  --data-path data/VOC201201 \
  --mask-encoding index \
  --num-classes 1 \
  --backbone-weights pretrained/resnet50.pth \
  --batch-size 32 --epochs 30 --workers 4 --device cuda --amp \
  --output run/train/vegann01_mpm5531_drmcbm_dynamic_bs32_ep30
```

显存不足降低batch；输出目录须为空。`--from-scratch`显式随机初始化，短流程测试建议epochs2、batch1。动态调制默认开启，`--no-dynamic-fusion`恢复原FEM用于对照；MPM保持固定。

当前版本断点恢复使用`--resume 运行目录/last.pth`、相同`--output`和全部原训练参数（包括总epochs），不需再次提供骨干权重。恢复优化器、调度器、AMP和随机状态。

## 自动结果与评估口径

训练/验证均有tqdm进度条，终端显示train/val loss、mIoU、Dice和耗时。二分类按验证集前景IoU、多分类按验证集mIoU保存best.pth；测试集不参与选模。结束后自动加载best.pth评估val与登记的test，保存：

| 文件 | 内容 |
|---|---|
| best.pth、last.pth | 最佳权重、完整训练恢复状态 |
| config.json、splits.json | 参数、数据配置与划分 |
| history.json、history.csv | 逐轮指标、学习率、耗时 |
| metrics.json、results.csv | 最佳轮次及验证/测试汇总 |
| class_metrics.csv | 每类IoU、Accuracy、Dice、Precision、Recall和真实像素数 |
| benchmark.json | 参数、计算量完整性、FPS和延迟 |
| loss_curve.png、metrics_curve.png | 训练曲线 |

参考[vegann-seg-benchmark](https://github.com/xuanpluspro/vegann-seg-benchmark)：区域指标累计整个数据集混淆矩阵；HD95、ASSD、Boundary F1逐图平均，距离单位像素，边界容差2像素。双方前景为空时距离0/F1为1，单方为空时距离取图像对角线/F1为0。区域指标未定义时null，类别宏平均排除未定义项，与参考仓库部分空类返回1的约定不同。

多分类mIoU包含背景，Dice/Precision/Recall取前景类别宏平均。Foreground IoU与边界指标衡量前景并集，不反映前景类别之间的混淆，应结合逐类IoU阅读。metric_units、aggregation记录单位和汇总方式。

没有test.txt时只输出验证结果，不冒充测试结果。`--no-benchmark`只关闭测速，仍导出精度；测速失败也保留已保存精度。

## 验证、汇总和预测

```bash
python val.py --data-path data/VOC201201 --weights run/train/你的实验/best.pth --split test
python summarize.py --data-path data/VOC201201 --weights run/train/你的实验/best.pth --device cuda --batch-size 32
python predict.py --data-path data/VOC201201/JPEGImages --weights run/train/你的实验/best.pth --output run/predict/你的实验
```

验证/汇总检查划分与训练一致。预测二分类默认输出0/255 PNG与H×W概率，可用`--mask-output-encoding index`导出0/1；多分类PNG为0..K-1，概率K×H×W，并输出彩色叠加图。输出目录须为空，不含大图滑窗。

## 参数量与测速

完整参数量以百万为单位。fvcore乘加计1次运算；存在unsupported_ops时gflops留空，counted_gflops仅为已统计部分，不能视为完整GFLOPs。

FPS固定batch1、512×512、PyTorch eager FP32，关闭autocast与TF32，默认预热50次、测量200次。CUDA Event计时只包含模型前向，不含读图、传输、预处理、后处理，不是TensorRT或端到端速度。CPU不生成GPU FPS。元数据记录设备和软件版本。

```bash
python benchmark.py --weights run/train/你的实验/best.pth --device cuda
```

## 测试与来源

```bash
python -m unittest discover -s tests -v
```

CPU合成数据测试覆盖512前向/反向、检查点恢复、动态融合开关、二分类/五分类训练、结果与预测导出；不能代替真实植物精度和CUDA验证。默认结构不保证比原方法更好。

MODA模块来源、分割改造与许可证见model/README.md及LICENSE.MODA；MPCNet来源与改写见model/MPM_SOURCE.md。当前5531是改写，不能称原版MPM严格复现。旧图和旧实验代码可在Git历史中查找。


## SACF：DRM替换SFA，CBM替换SDE

三个自顶向下位置均改为：深层[B,C,h,w]→DRM→上采样到[2h,2w]；浅层[B,C,2h,2w]→CBM；再使用原有自适应融合，输出[B,C,2h,2w]。三个位置不共享参数。MPM-5531、三个CAFR动态调制、损失、标签编码和指标口径未改。

DRM用两路3×3普通卷积（dilation=1/2），直接从输入经GAP生成两个图像级Softmax权重，逐分支加权后Concat并用1×1融合，输出x+gamma*fused；gamma初始化0。CBM采用DWConv→PWConv→GELU局部分支，加上GAP→Conv→Sigmoid生成的通道权重对原输入的重加权，经过输出投影再加输入。没有导入完整HFCB的BN、两路平均或额外外层gamma；来源与改写见model/SACF_SOURCE.md。

检查点格式更新为v3；旧完整模型不能resume到新结构，请从相同ResNet骨干预训练开始新实验。旧权重和结果文件不会被删除，旧代码保留在Git历史。当前未加入QKV。
