# MaskDINO–SAR 弱对齐建筑实例高度 Baseline

本实现以 MaskDINO 的建筑 query、最终 reference box 和 stride-8 光学特征为入口，从配对的四通道 SAR 中提取多尺度信息，经过全局偏移、实例局部相关偏移和三层 Multi-Scale Deformable Cross-Attention 后直接回归每个建筑实例的高度。

## 本机数据与环境

当前已验证配置：

```text
Python: /home/lzq/miniconda3/envs/dino/bin/python
PyTorch: 2.7.0+cu128
GPU: NVIDIA GeForce RTX 5070
数据: /run/media/lzq/LZQ/sp/SpaceNet6_Expanded/PS-RGB/sar_rgb_dataset_1m_256up512
标注: building_height_coco_repaired.json
```

数据直接从原目录读取，不复制 16GB 影像。代码按固定 seed 将单个 COCO 文件划分为 80%/10%/10% 的 train/val/test。当前划分的 val 为 465 张、test 为 424 张。标注中 24 个无有效高度的实例仍可参与实例分割匹配，但会由 `gt_height_valid` 从高度损失中排除。

SAR GeoTIFF 的物理通道顺序为：

```text
sin(phi), cos(phi), coherence, intensity_db
```

OpenCV 对该数据的前三个 band 读取顺序是反向的，因此配置中的 `INPUT.SAR_CHANNEL_ORDER: [2, 1, 0, 3]` 不应删除。

## 服务器迁移时只改这里

打开主配置：

```text
configs/sp6/instance-segmentation/maskdino_R50_bs2_50ep_height.yaml
```

文件顶部的 `PATHS` 是所有需要随机器调整的位置：

```yaml
PATHS:
  DATASET_ROOT: ...
  ANNOTATION_FILE: ...
  MASKDINO_WEIGHTS: ...
  SAR_RESNET18_WEIGHTS: ...
  OUTPUT_DIR: ...
```

下面的配置项通过 YAML alias 自动引用这些值，不需要重复修改。数据根目录和标注也可临时由环境变量覆盖：

```bash
export SP6_DATASET_ROOT=/server/path/sar_rgb_dataset_1m_256up512
export SP6_ANNOTATION_FILE=/server/path/sar_rgb_dataset_1m_256up512/building_height_coco_repaired.json
```

`MASKDINO_WEIGHTS` 当前指向本机已有的 SP6 光学模型 `model_0077279.pth`，checkpoint 不纳入 Git。迁移服务器时需要一并复制该文件并修改上述路径。`SAR_RESNET18_WEIGHTS` 默认使用 PyTorch 官方 URL；如果服务器不能联网，先下载 `resnet18-f37072fd.pth`，再把该项改成本地绝对路径。

## 数据和模块验收

先激活已有环境：

```bash
conda activate dino
```

检查数据配对、通道、相位向量和高度字段：

```bash
python tools/validate_sp6_sar_height_dataset.py --samples 16
```

检查 39 通道 Phase Descriptor、偏移符号、FPN/Decoder 形状、匹配高度损失、query 索引同步以及端到端梯度：

```bash
python -m unittest -v tests.test_sar_height
```

预期为 8 项测试全部 `OK`。其中合成平移测试明确要求 SAR 向右移动 8 个原始像素时预测 `dx` 为正；另有独立测试确认关闭 SAR 后 optical class、box 和 mask 与启用前逐元素一致。

## 小样本过拟合检查

过拟合配置固定使用两张训练图，不做随机翻转或 synthetic shift：

```bash
python train_net.py \
  --num-gpus 1 \
  --config-file configs/sp6/instance-segmentation/maskdino_R50_sar_height_overfit.yaml
```

该配置用于确认高度损失可以下降，以及梯度能传递到 Height Head、Geometry Decoder、Deformable Attention、SAR FPN、ResNet18、Phase Stem 和 Amplitude Stem；不要把它用于正式结果。

本机已完成 500 iteration 验收：前 5 个日志点的高度损失中位数为 `0.7234`，后 5 个为 `0.0685`，下降约 `90.5%`。测试 checkpoint 和日志均位于被 Git 忽略的 `output/sp6_sar_height_overfit/`。

## 正式训练

```bash
python train_net.py \
  --num-gpus 1 \
  --config-file configs/sp6/instance-segmentation/maskdino_R50_bs2_50ep_height.yaml
```

默认冻结 MaskDINO，只训练 SAR Encoder、Matching Projection、全局/局部相关模块、Geometry Decoder 和 Height Head。训练损失为匹配正样本的 Smooth L1 高度损失，加上已知 synthetic shift 的全局偏移损失。Matcher cost 仍只包含 class、box 和 mask，不含高度。

断点续训：

```bash
python train_net.py \
  --resume \
  --num-gpus 1 \
  --config-file configs/sp6/instance-segmentation/maskdino_R50_bs2_50ep_height.yaml
```

## 验证与推理

```bash
python train_net.py \
  --eval-only \
  --num-gpus 1 \
  --config-file configs/sp6/instance-segmentation/maskdino_R50_bs2_50ep_height.yaml \
  MODEL.WEIGHTS /path/to/model_final.pth
```

`Instances` 输出字段包括：

```text
pred_masks
pred_boxes
scores
pred_classes
pred_query_indices
pred_heights
```

`pred_query_indices` 明确保留 postprocessor 选中的 query，因此 class、box、mask 与 height 始终来自同一个 query。每张图的结果还包含 `global_sar_offset` 和所有 query 的 `local_sar_offsets`，便于检查弱对齐行为。评估输出同时包含 COCO bbox/segm AP，以及高度 MAE、RMSE、Bias、Matched 和 Coverage。

## 关键退化开关

主配置支持需求文档中的全部退化路径：

```text
MODEL.SAR.AMP_ENABLED=False           -> 只使用 Phase Stem
MODEL.SAR.AMP_GATE_ENABLED=False      -> Phase + Amplitude 直接相加
MODEL.ALIGN.GLOBAL_ENABLED=False      -> global offset 为 0
MODEL.ALIGN.LOCAL_CORR_ENABLED=False  -> local offset 为 0
二者均关闭                              -> SAR reference box 与 optical reference box 完全相同
MODEL.SAR.ENABLED=False               -> 不创建/执行 SAR 分支，原 class/box/mask 路径保持不变
```

完整结构与非目标以仓库根目录的开发需求文档为准。
