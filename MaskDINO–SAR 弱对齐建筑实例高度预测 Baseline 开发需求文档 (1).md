# MaskDINO–SAR 弱对齐建筑实例高度预测 Baseline 开发需求文档

## 1. 文档目标

本文档定义一个基于现有 MaskDINO 源代码开发的弱对齐 SAR–光学建筑实例高度预测 Baseline。光学分支继续由 MaskDINO 完成建筑实例检测与分割；SAR 分支从四通道干涉数据中提取多尺度特征；随后通过“全局偏移、实例局部相关偏移、Deformable Attention 残差偏移”三级位置修正，将 SAR 信息注入 MaskDINO 的建筑 query；最终直接从更新后的多模态 query 回归建筑高度。

该版本以结构清晰、可实现和便于调试为优先目标。Baseline 中不引入建筑内部、边界和周围环带参考点，不进行相位解缠，不增加 optical-only height prior，也不把最终高度参数化为 SAR 高度残差。MaskDINO 原始 query 只作为建筑实例表示和 SAR 检索条件，最终高度由经过三层 SAR Deformable Cross-Attention 更新后的 query 直接预测。

---

## 2. 总体网络结构

整体数据流如下：

```text
Optical Image
      │
      ▼
   MaskDINO
      │
      ├── class logits
      ├── boxes
      ├── instance masks
      ├── final decoder queries
      └── stride-8 optical feature
                         │
                         │
                         ▼
                  Geometry Query
                         │
                         │
SAR 4 Channels          │
sinφ cosφ γ Idb         │
      │                  │
      ▼                  │
SAR Preprocessing        │
      │                  │
      ├── Xphase: 39 ch  │
      └── Xamp: 2 ch     │
      │                  │
      ▼                  │
Lightweight Dual Stem    │
      │                  │
      ▼                  │
Gated Early Fusion       │
      │                  │
      ▼                  │
One Shared ResNet18      │
      │                  │
      ▼                  │
SAR FPN: Fs              │
      │                  │
      ├── Global Offset  │
      ├── Local Corr.    │
      └── SAR Memory     │
             │           │
             └─────┬─────┘
                   ▼
        3-layer Deformable
         Cross-Attention
                   │
                   ▼
       Multimodal Geometry Query
                   │
                   ▼
              Height Head
                   │
                   ▼
            Building Height
```

核心关系为：

$$
Q_{\text{MaskDINO}}
\rightarrow
Q_{\text{geometry}}^{0}
\rightarrow
Q_{\text{geometry}}^{3}
\rightarrow
\hat{H}
$$

其中，三层 Geometry Decoder 通过 SAR 多尺度特征不断更新 query：

$$
Q_{\text{geometry}}^{t+1}
=
\text{GeometryDecoderLayer}
\left(
Q_{\text{geometry}}^{t},
F_s
\right)
$$

最终：

$$
\hat{H}
=
\text{HeightHead}
\left(
Q_{\text{geometry}}^{3}
\right)
$$

---

## 3. Baseline 范围与非目标

本版本只实现以下主线：

1. MaskDINO 提供建筑实例 mask、box、query 和光学低层特征。
2. SAR 四通道经过简单双 Stem、门控融合、单个 ResNet18-FPN 得到多尺度特征。
3. 使用光学和 SAR 的 stride-8 特征估计 tile 级全局偏移。
4. 使用建筑 query 和 reference box 中心估计实例级局部相关偏移。
5. 使用原始 Multi-Scale Deformable Attention 预测剩余采样偏移。
6. 三层 Geometry Decoder 将 SAR 信息写入建筑 query。
7. 最终 query 直接预测建筑高度。

本版本明确不实现：

- 建筑 interior、boundary、ring 参考点；
- 人工结构参考点采样；
- 相位解缠；
- optical-only height head；
- optical height 与 SAR residual height 相加；
- 两套完整的 ResNet18；
- SAR 反向更新 MaskDINO mask 和 box；
- 单独的物理建筑 offset head；
- 复杂的配准 flow 或非刚性形变场。

---

# 4. 输入与输出定义

## 4.1 网络输入

光学输入：

$$
O
\in
\mathbb{R}^{B\times C_o\times H\times W}
$$

SAR 原始输入固定为四通道：

$$
S
=
[
\sin\phi,
\cos\phi,
\gamma,
I_{dB}
]
$$

对应张量：

$$
S
\in
\mathbb{R}^{B\times4\times H\times W}
$$

通道顺序必须固定为：

```text
channel 0: sin(phi)
channel 1: cos(phi)
channel 2: coherence
channel 3: intensity_db
```

每个训练样本的 target 至少需要包含：

```python
target = {
    "labels": Tensor[N],
    "boxes": Tensor[N, 4],
    "masks": Tensor[N, H, W],
    "heights": Tensor[N],
}
```

`heights[i]` 必须与 `labels[i]`、`boxes[i]` 和 `masks[i]` 表示同一个建筑实例。

---

## 4.2 网络输出

网络保留 MaskDINO 原始输出，并增加：

```python
outputs = {
    "pred_logits": ...,
    "pred_boxes": ...,
    "pred_masks": ...,
    "pred_heights": ...,
    "global_sar_offset": ...,
    "local_sar_offsets": ...,
}
```

其中：

```text
pred_heights:
[B, N_query]
```

每一个 MaskDINO query 对应一个高度预测，但高度损失只计算在 Hungarian Matcher 匹配到真实建筑的正样本 query 上。

---

# 5. 光学分支与 MaskDINO 接口要求

MaskDINO 主体结构保持不变。需要修改其 forward 接口，使其额外暴露三个张量：

```python
maskdino_features = {
    "decoder_query": final_query,
    "reference_boxes": reference_boxes,
    "optical_feature_s8": optical_feature_s8,
}
```

`decoder_query` 为 MaskDINO 最后一层 decoder 的 query content：

$$
Q_o
\in
\mathbb{R}^{B\times N_q\times D_o}
$$

`reference_boxes` 为对应 query 的归一化 reference box：

$$
B_o
=
(c_x,c_y,w,h)
$$

其值域应为：

$$
c_x,c_y,w,h\in[0,1]
$$

优先使用 MaskDINO/DINO decoder 中已有的最终 reference box。如果当前源代码没有直接返回该张量，可以使用最终预测 box 构造 reference box。

`optical_feature_s8` 为 pixel decoder 输出的 stride-8 光学特征：

$$
F_o^8
\in
\mathbb{R}^{B\times C_o^8\times H/8\times W/8}
$$

该特征仅用于全局偏移和局部相关计算，不直接进入 Height Head。

Geometry Query 使用 MaskDINO query 初始化：

$$
Q_g^0
=
\text{LayerNorm}
\left(
W_q Q_o
\right)
$$

其中：

$$
Q_g^0
\in
\mathbb{R}^{B\times N_q\times256}
$$

如果 MaskDINO 的 query 维度已经是 256，仍建议保留独立的线性投影和 LayerNorm，使 Geometry Decoder 与原始 MaskDINO decoder 参数解耦。

Baseline 中不增加 optical height head。高度监督只施加在三层 SAR Geometry Decoder 之后。

默认开发配置中，MaskDINO 参数冻结，reference box 在进入 SAR 模块前执行 detach：

```python
q_geo = geometry_query_proj(maskdino_query.detach())
reference_boxes = reference_boxes.detach()
```

后续可通过配置开关允许联合微调。

---

# 6. SAR 输入预处理

## 6.1 Phase Descriptor

Phase 输入在进入网络前构造成 39 通道：

$$
X_{\text{phase}}
\in
\mathbb{R}^{B\times39\times H\times W}
$$

首先保留三个基础通道：

$$
\sin\phi,\quad
\cos\phi,\quad
\gamma
$$

然后使用三个半径：

$$
r\in\{1,2,4\}
$$

每个半径使用四个方向：

$$
(r,0),\quad
(0,r),\quad
(r,r),\quad
(r,-r)
$$

因此一共有 12 个邻域偏移：

$$
\delta_1,\delta_2,\ldots,\delta_{12}
$$

对于每一个偏移，计算：

$$
D_c^\delta(p)
=
\cos\phi(p)\cos\phi(p+\delta)
+
\sin\phi(p)\sin\phi(p+\delta)
$$

$$
D_s^\delta(p)
=
\sin\phi(p)\cos\phi(p+\delta)
-
\cos\phi(p)\sin\phi(p+\delta)
$$

以及 pairwise coherence：

$$
R^\delta(p)
=
\sqrt{
\gamma(p)\gamma(p+\delta)
}
$$

最终：

$$
X_{\text{phase}}
=
[
\sin\phi,
\cos\phi,
\gamma,
D_s^{\delta_1},
D_c^{\delta_1},
R^{\delta_1},
\ldots,
D_s^{\delta_{12}},
D_c^{\delta_{12}},
R^{\delta_{12}}
]
$$

通道数为：

$$
3+12\times3=39
$$

Phase Descriptor Generator 不包含可学习参数。

不要使用 `torch.roll` 直接生成邻域通道，因为它会把图像右边界循环到左边界。建议使用显式 padding 和 slicing，或者 `grid_sample`。超出图像范围的位置设置：

```text
Ds = 0
Dc = 0
R  = 0
```

其中 `R=0` 表示该邻域关系无效。

如果数据增强中对 `sinφ` 和 `cosφ` 使用了插值，在生成 39 通道之前应重新归一化：

$$
r(p)
=
\sqrt{
\sin^2\phi(p)+\cos^2\phi(p)+\epsilon
}
$$

$$
\sin\phi(p)
\leftarrow
\frac{\sin\phi(p)}{r(p)}
$$

$$
\cos\phi(p)
\leftarrow
\frac{\cos\phi(p)}{r(p)}
$$

所有几何增强和可选的 synthetic SAR shift 都应先作用于原始四通道 SAR，之后再生成 39 通道 Phase Descriptor。

---

## 6.2 Amplitude 输入

Amplitude 输入保持简单，只使用归一化强度和 coherence：

$$
X_{\text{amp}}
=
[
I_n,
\gamma
]
$$

其中：

$$
X_{\text{amp}}
\in
\mathbb{R}^{B\times2\times H\times W}
$$

$I_{dB}$ 不再进行对数变换，只做训练集统计意义上的 clipping 和标准化：

$$
I_n
=
\frac{
\text{Clip}(I_{dB})-\mu_I
}{
\sigma_I+\epsilon
}
$$

建议使用训练集统一的均值和标准差，不采用逐 tile 标准化，以免完全消除 tile 之间可能有意义的强度差异。

该 Baseline 不计算强度梯度、局部均值、局部方差或其他物理派生特征。

---

# 7. SAR Lightweight Stem 与共享 ResNet18

## 7.1 Phase Stem

Phase Stem 输入 39 通道，输出 stride-4、64 通道特征。

```text
Xphase
[B, 39, H, W]

    │ Conv 3×3, stride 2, 39 → 32
    │ GroupNorm
    │ ReLU

[B, 32, H/2, W/2]

    │ Conv 3×3, stride 2, 32 → 64
    │ GroupNorm
    │ ReLU

Fphase
[B, 64, H/4, W/4]
```

对应：

$$
F_{\text{phase}}
=
\text{PhaseStem}
\left(
X_{\text{phase}}
\right)
$$

---

## 7.2 Amplitude Stem

Amplitude Stem 输入 2 通道，保持较小容量：

```text
Xamp
[B, 2, H, W]

    │ Conv 3×3, stride 2, 2 → 16
    │ GroupNorm
    │ ReLU

[B, 16, H/2, W/2]

    │ Conv 3×3, stride 2, 16 → 32
    │ GroupNorm
    │ ReLU

[B, 32, H/4, W/4]

    │ Conv 1×1, 32 → 64

Famp
[B, 64, H/4, W/4]
```

对应：

$$
F_{\text{amp}}
=
\text{AmpStem}
\left(
X_{\text{amp}}
\right)
$$

Amplitude Stem 只负责生成轻量辅助特征，不包含独立 backbone。

---

## 7.3 门控融合

Phase 和 Amplitude 特征已经具有相同的分辨率和通道数：

$$
F_{\text{phase}},
F_{\text{amp}}
\in
\mathbb{R}^{B\times64\times H/4\times W/4}
$$

首先计算 gate：

$$
G_{\text{amp}}
=
\text{Sigmoid}
\left(
\text{Conv}_{1\times1}
[
F_{\text{phase}},F_{\text{amp}}
]
\right)
$$

其中：

```text
Concat:
64 + 64 = 128 channels

Gate Conv:
128 → 64 channels
```

融合特征为：

$$
F_{\text{stem}}^0
=
F_{\text{phase}}
+
G_{\text{amp}}
\odot
F_{\text{amp}}
$$

随后使用一个简单的融合卷积：

```text
Conv 3×3, 64 → 64
GroupNorm
ReLU
```

得到：

$$
F_{\text{stem}}
\in
\mathbb{R}^{B\times64\times H/4\times W/4}
$$

整个双 Stem 和 Gate 只作为 ResNet18 之前的浅层输入投影。网络中始终只有一套深层 ResNet18。

---

## 7.4 共享 ResNet18

删除标准 ResNet18 的以下部分：

```text
conv1
bn1
relu
maxpool
```

将 $F_{\text{stem}}$ 直接输入 `layer1`：

$$
C_2
=
\text{layer1}
\left(
F_{\text{stem}}
\right)
$$

各 stage 输出为：

| 特征 | 通道数 | 相对输入步长 |
|---|---:|---:|
| $C_2$ | 64 | 4 |
| $C_3$ | 128 | 8 |
| $C_4$ | 256 | 16 |
| $C_5$ | 512 | 32 |

即：

$$
C_2
\in
\mathbb{R}^{B\times64\times H/4\times W/4}
$$

$$
C_3
\in
\mathbb{R}^{B\times128\times H/8\times W/8}
$$

$$
C_4
\in
\mathbb{R}^{B\times256\times H/16\times W/16}
$$

$$
C_5
\in
\mathbb{R}^{B\times512\times H/32\times W/32}
$$

`layer1-layer4` 可以加载 ImageNet ResNet18 预训练权重。考虑检测任务通常 batch 较小，建议使用 Frozen BatchNorm，或者保持与当前 MaskDINO 工程一致的归一化策略。

---

## 7.5 SAR FPN

使用标准 FPN 将四个 stage 全部投影到 256 通道：

$$
P_2,P_3,P_4,P_5
$$

具体尺寸为：

| 特征 | 通道数 | 步长 |
|---|---:|---:|
| $P_2$ | 256 | 4 |
| $P_3$ | 256 | 8 |
| $P_4$ | 256 | 16 |
| $P_5$ | 256 | 32 |

最终定义：

$$
F_s
=
\{P_2,P_3,P_4,P_5\}
$$

$F_s$ 是 Geometry Decoder 的 SAR memory。后续 Deformable Cross-Attention 直接从这四个 level 中采样。

---

# 8. 光学–SAR Matching Feature

全局偏移和局部相关不直接使用完整的 256 维 FPN 特征，而使用独立的低维 matching projection。

光学侧：

$$
E_o
=
\text{L2Norm}
\left(
\text{Conv}_{1\times1}
(F_o^8)
\right)
$$

SAR 侧：

$$
E_s
=
\text{L2Norm}
\left(
\text{Conv}_{1\times1}
(P_3)
\right)
$$

两侧都投影成 64 通道：

$$
E_o,E_s
\in
\mathbb{R}^{B\times64\times H/8\times W/8}
$$

如果 MaskDINO 的 stride-8 feature 与 $P_3$ 空间尺寸不同，应通过双线性插值统一到同一尺寸。

Matching Feature 与最终 Deformable Attention value feature 分开。相关模块使用 $E_o$ 和 $E_s$，Geometry Decoder 读取的是完整的 $F_s$。

---

# 9. Tile 级全局偏移

全局偏移用于估计整张 SAR tile 相对于光学 tile 的共享平移：

$$
d_g
=
(d_x^g,d_y^g)
$$

在 stride-8 matching feature 上设置候选位移：

$$
\delta
\in
[-R_g,R_g]^2
$$

对于每一个候选位移，计算全局相关性：

$$
C_g(\delta)
=
\frac{1}{|\Omega_\delta|}
\sum_{x\in\Omega_\delta}
E_o(x)^\mathsf{T}
E_s(x+\delta)
$$

越界位置不参与平均。

将相关体转换为概率：

$$
P_g(\delta)
=
\text{Softmax}
\left(
\frac{C_g(\delta)}{\tau_g}
\right)
$$

使用 soft-argmax 得到 stride-8 坐标中的全局偏移：

$$
\hat{\delta}_g
=
\sum_{\delta}
P_g(\delta)\delta
$$

转换到归一化图像坐标：

$$
d_x^g
=
\frac{8\hat{\delta}_{g,x}}{W}
$$

$$
d_y^g
=
\frac{8\hat{\delta}_{g,y}}{H}
$$

其中正的 $d_x^g$ 表示 SAR 对应位置位于光学位置的右侧，正的 $d_y^g$ 表示 SAR 对应位置位于光学位置的下方。

全局偏移每张图像只预测一次，并由该图像中的所有 query 共享。

---

# 10. 实例级 Local Correlation

Local Correlation 为每一个 query 预测独立的局部偏移：

$$
d_i^{local}
=
(d_{i,x}^{local},d_{i,y}^{local})
$$

该模块不使用 interior、boundary 或 ring，不生成额外人工 reference points。每个 query 只使用 MaskDINO 原始 reference box 的中心：

$$
c_i
=
(c_{i,x},c_{i,y})
$$

首先构造 query-conditioned optical template：

$$
t_i
=
\text{L2Norm}
\left(
W_t q_i^0
+
\text{Sample}(E_o,c_i)
\right)
$$

其中：

$$
t_i\in\mathbb{R}^{64}
$$

`Sample` 使用双线性插值。

经过全局偏移后，SAR 搜索中心为：

$$
c_i^g
=
c_i+d_g
$$

在 stride-8 matching feature 上，以 $c_i^g$ 为中心搜索：

$$
\delta
\in
[-R_l,R_l]^2
$$

每一个候选位置的 SAR 向量为：

$$
s_i(\delta)
=
\text{Sample}
\left(
E_s,
c_i^g+\delta
\right)
$$

局部相关性为：

$$
C_i^{local}(\delta)
=
t_i^\mathsf{T}s_i(\delta)
$$

然后：

$$
P_i^{local}(\delta)
=
\text{Softmax}
\left(
\frac{
C_i^{local}(\delta)
}{
\tau_l
}
\right)
$$

局部位移为：

$$
\hat{\delta}_i^{local}
=
\sum_{\delta}
P_i^{local}(\delta)\delta
$$

转换到归一化图像坐标：

$$
d_{i,x}^{local}
=
\frac{
8\hat{\delta}_{i,x}^{local}
}{
W
}
$$

$$
d_{i,y}^{local}
=
\frac{
8\hat{\delta}_{i,y}^{local}
}{
H
}
$$

Local Correlation 对所有 query 计算，但高度损失仍只施加在匹配到真实建筑的 query 上。

Baseline 中，全局偏移和局部偏移在进入三层 Geometry Decoder 前各计算一次，三层 decoder 共享相同的结果，不在每一层重复计算 correlation。

---

# 11. SAR Reference Box 构造

MaskDINO 输出的 optical reference box 为：

$$
b_i^o
=
(c_{i,x},c_{i,y},w_i,h_i)
$$

SAR reference box 只修改中心，不修改宽高：

$$
b_i^s
=
(
c_{i,x}+d_x^g+d_{i,x}^{local},
c_{i,y}+d_y^g+d_{i,y}^{local},
w_i,
h_i
)
$$

也就是说：

$$
b_i^s
=
b_i^o
+
d_g
+
d_i^{local}
$$

其中偏移只作用在前两个坐标。

实现时可以将中心限制在合理范围：

```python
sar_center = sar_center.clamp(0.0, 1.0)
```

但 correlation 搜索阶段应使用有效区域 mask，而不是简单把所有越界候选位置截断到边界。

---

# 12. 三层 Deformable Geometry Decoder

Geometry Decoder 使用三层 Multi-Scale Deformable Cross-Attention：

```text
q_geo_0
   │
   ▼
Geometry Decoder Layer 1
   │
   ▼
q_geo_1
   │
   ▼
Geometry Decoder Layer 2
   │
   ▼
q_geo_2
   │
   ▼
Geometry Decoder Layer 3
   │
   ▼
q_geo_3
```

每层默认配置：

```text
d_model  = 256
n_heads  = 8
n_levels = 4
n_points = 4
```

四个 feature level 为：

$$
F_s
=
\{P_2,P_3,P_4,P_5\}
$$

Baseline 应尽量复用 MaskDINO 或 Deformable DETR 源代码中现有的 `MSDeformAttn` 实现，不修改 CUDA 算子。三级偏移通过修改 reference box 中心实现：

1. 全局偏移写入 SAR reference box；
2. Local Correlation 偏移写入 SAR reference box；
3. Deformable residual offset 继续由原始 `MSDeformAttn` 内部预测。

因此实际关系为：

$$
\text{SAR reference center}
=
c_i
+
d_g
+
d_i^{local}
$$

在此基础上，标准 Multi-Scale Deformable Attention 预测：

$$
\Delta p_{i,h,l,k}^{t}
$$

并按照原始 4D reference box 规则生成采样位置。概念上可以写成：

$$
p_{i,h,l,k}^{t}
=
c_i
+
d_g
+
d_i^{local}
+
d_{i,h,l,k}^{deform,t}
$$

其中：

- $d_g$：每张图像共享；
- $d_i^{local}$：每个 query 独立；
- $d_{i,h,l,k}^{deform,t}$：每层、每个 head、每个 level、每个 sampling point 独立。

这里的 Deformable Offset 不再承担完整跨模态配准，只负责在已经完成全局和局部修正的区域内寻找更有价值的 SAR 特征。

---

## 12.1 Geometry Decoder Layer

为保持 Baseline 轻量，默认每层只包含：

```text
Deformable Cross-Attention
Residual + LayerNorm
FFN
Residual + LayerNorm
```

默认不再增加 query self-attention，因为 MaskDINO query 已经经过完整 decoder 建模。需要时可以通过配置开关启用。

每层更新为：

$$
Z_i^t
=
\text{MSDeformAttn}
\left(
q_i^t,
F_s,
b_i^s
\right)
$$

$$
\tilde q_i^{t+1}
=
\text{LayerNorm}
\left(
q_i^t+Z_i^t
\right)
$$

$$
q_i^{t+1}
=
\text{LayerNorm}
\left(
\tilde q_i^{t+1}
+
\text{FFN}
(\tilde q_i^{t+1})
\right)
$$

经过三层以后：

$$
Q_g^3
$$

被视为已经融合 SAR 信息的 multimodal geometry query。

Baseline 中不根据 Geometry Decoder 输出重新预测 mask、class 或 box，也不进行 reference box iterative refinement。

---

# 13. SAR Memory 的标准展开

为了复用 MSDeformAttn，需要按照 Deformable DETR 的方式处理 $F_s$。

对每个 FPN level 加入二维位置编码和 level embedding：

$$
\tilde P_l
=
P_l
+
PE_l
+
E_l^{level}
$$

然后依次 flatten：

```python
src_flatten
spatial_shapes
level_start_index
valid_ratios
padding_mask
```

这些张量格式应与当前 MaskDINO 中的 Deformable Attention 实现保持一致。

SAR FPN 的输出维度固定为 256，因此不需要在 Geometry Decoder 前再次进行 value projection。

---

# 14. Height Head

最终高度只从三层 decoder 后的 query 中预测：

$$
\hat h_i
=
\text{Softplus}
\left(
\text{MLP}_{height}
(q_i^3)
\right)
$$

Height Head 建议使用：

```text
LayerNorm
Linear 256 → 256
ReLU
Dropout
Linear 256 → 1
Softplus
```

输出：

$$
\hat H
\in
\mathbb{R}^{B\times N_q}
$$

Baseline 不使用：

$$
h^{opt}+\Delta h^{sar}
$$

也不在 SAR backbone 上单独预测高度。SAR 信息先进入 query，Height Head 再从更新后的 query 直接回归最终高度。

---

# 15. Matcher 与高度损失

继续使用 MaskDINO 原始 Hungarian Matcher 完成预测 query 与 GT 实例的匹配。匹配 cost 仍然只由原始分类、box 和 mask cost 构成，不增加高度 cost。

假设 matcher 输出：

```python
indices = [
    (src_idx_0, tgt_idx_0),
    (src_idx_1, tgt_idx_1),
    ...
]
```

则高度预测和 GT 高度按照相同索引提取：

```python
pred_height = outputs["pred_heights"][batch_idx, src_idx]
gt_height = targets[batch_idx]["heights"][tgt_idx]
```

高度损失使用 Smooth L1：

$$
L_h
=
\frac{1}{N_+}
\sum_i
\text{SmoothL1}
\left(
\hat h_i,h_i
\right)
$$

总损失为：

$$
L
=
L_{\text{MaskDINO}}
+
\lambda_h L_h
$$

如果 MaskDINO 完全冻结，可以只计算：

$$
L=L_h
$$

如果启用 synthetic global shift supervision，则额外增加：

$$
L
=
L_{\text{MaskDINO}}
+
\lambda_hL_h
+
\lambda_gL_g
$$

Synthetic shift 只用于稳定全局偏移模块，不要求为 Local Correlation 和 Deformable Offset 提供单独监督。

---

# 16. 推荐 Forward 流程

```python
def forward(optical, sar_4ch, targets=None):
    # --------------------------------------------------
    # 1. MaskDINO optical branch
    # --------------------------------------------------
    optical_out = self.maskdino(
        optical,
        return_decoder_query=True,
        return_reference_boxes=True,
        return_stride8_feature=True,
    )

    q_mask = optical_out["decoder_query"]
    ref_boxes_opt = optical_out["reference_boxes"]
    optical_s8 = optical_out["optical_feature_s8"]

    if self.freeze_maskdino:
        q_mask = q_mask.detach()
        ref_boxes_opt = ref_boxes_opt.detach()
        optical_s8 = optical_s8.detach()

    q_geo = self.geometry_query_proj(q_mask)

    # --------------------------------------------------
    # 2. SAR preprocessing
    # --------------------------------------------------
    sin_phi = sar_4ch[:, 0:1]
    cos_phi = sar_4ch[:, 1:2]
    coherence = sar_4ch[:, 2:3]
    intensity_db = sar_4ch[:, 3:4]

    x_phase = self.phase_descriptor_generator(
        sin_phi,
        cos_phi,
        coherence,
    )

    intensity_norm = self.intensity_normalizer(
        intensity_db
    )

    x_amp = torch.cat(
        [intensity_norm, coherence],
        dim=1,
    )

    # --------------------------------------------------
    # 3. SAR encoder
    # --------------------------------------------------
    f_phase = self.phase_stem(x_phase)
    f_amp = self.amp_stem(x_amp)

    amp_gate = torch.sigmoid(
        self.amp_gate(
            torch.cat([f_phase, f_amp], dim=1)
        )
    )

    f_stem = f_phase + amp_gate * f_amp
    f_stem = self.fusion_conv(f_stem)

    c2 = self.sar_resnet.layer1(f_stem)
    c3 = self.sar_resnet.layer2(c2)
    c4 = self.sar_resnet.layer3(c3)
    c5 = self.sar_resnet.layer4(c4)

    p2, p3, p4, p5 = self.sar_fpn(
        c2, c3, c4, c5
    )

    fs = [p2, p3, p4, p5]

    # --------------------------------------------------
    # 4. Matching features
    # --------------------------------------------------
    e_opt = self.optical_match_proj(optical_s8)
    e_sar = self.sar_match_proj(p3)

    e_opt = l2_normalize(e_opt, dim=1)
    e_sar = l2_normalize(e_sar, dim=1)

    # --------------------------------------------------
    # 5. Global offset
    # --------------------------------------------------
    if self.use_global_offset:
        global_offset = self.global_correlation(
            e_opt,
            e_sar,
        )
    else:
        global_offset = zeros_global_offset(
            optical.shape[0]
        )

    # --------------------------------------------------
    # 6. Per-query local correlation
    # --------------------------------------------------
    if self.use_local_correlation:
        local_offsets = self.local_correlation(
            q_geo,
            ref_boxes_opt,
            e_opt,
            e_sar,
            global_offset,
        )
    else:
        local_offsets = zeros_local_offsets(
            q_geo
        )

    # --------------------------------------------------
    # 7. Build SAR reference boxes
    # --------------------------------------------------
    ref_boxes_sar = shift_reference_boxes(
        ref_boxes_opt,
        global_offset,
        local_offsets,
    )

    # --------------------------------------------------
    # 8. Three-layer geometry decoder
    # --------------------------------------------------
    q_geo = self.geometry_decoder(
        query=q_geo,
        reference_boxes=ref_boxes_sar,
        multi_scale_features=fs,
    )

    # --------------------------------------------------
    # 9. Height prediction
    # --------------------------------------------------
    pred_heights = self.height_head(
        q_geo
    ).squeeze(-1)

    outputs = {
        **optical_out,
        "pred_heights": pred_heights,
        "global_sar_offset": global_offset,
        "local_sar_offsets": local_offsets,
    }

    return outputs
```

---

# 17. 配置开关

Baseline 应提供以下配置开关。开关主要用于调试模块和控制退化路径，不要求围绕这些开关设计复杂消融实验。

| 配置项 | 默认值 | 作用 |
|---|---:|---|
| `MODEL.SAR.ENABLED` | `True` | 是否启用整个 SAR 分支 |
| `MODEL.SAR.PHASE_DESCRIPTOR` | `True` | 是否使用 39 通道 phase descriptor |
| `MODEL.SAR.AMP_ENABLED` | `True` | 是否使用 $I_{dB}+\gamma$ 的轻量 Amp Stem |
| `MODEL.SAR.AMP_GATE_ENABLED` | `True` | 是否使用门控融合 |
| `MODEL.ALIGN.GLOBAL_ENABLED` | `True` | 是否启用 tile 级全局偏移 |
| `MODEL.ALIGN.LOCAL_CORR_ENABLED` | `True` | 是否启用实例级 Local Correlation |
| `MODEL.GEOMETRY_DECODER.NUM_LAYERS` | `3` | Geometry Decoder 层数 |
| `MODEL.GEOMETRY_DECODER.USE_SELF_ATTN` | `False` | 是否在每层加入 query self-attention |
| `MODEL.MASKDINO.FREEZE` | `True` | 是否冻结 MaskDINO |
| `MODEL.MASKDINO.DETACH_REFERENCE` | `True` | SAR 分支是否阻断对 box/reference 的梯度 |
| `MODEL.ALIGN.SYNTHETIC_SHIFT_SUPERVISION` | `True` | 是否使用已知随机平移监督 global offset |
| `MODEL.HEIGHT.ACTIVATION` | `"softplus"` | 高度输出激活方式 |

对应退化关系应明确实现：

```text
AMP_ENABLED = False:
    Fstem = Fphase

AMP_GATE_ENABLED = False:
    Fstem = Fphase + Famp

GLOBAL_ENABLED = False:
    d_global = 0

LOCAL_CORR_ENABLED = False:
    d_local = 0

GLOBAL_ENABLED = False
and
LOCAL_CORR_ENABLED = False:
    Deformable Attention 直接围绕 Optical reference box 采样
```

搜索范围不是布尔开关，但需要放入配置文件：

```yaml
MODEL:
  ALIGN:
    MATCHING_STRIDE: 8
    GLOBAL_SEARCH_RADIUS: 8
    LOCAL_SEARCH_RADIUS: 4
    GLOBAL_TEMPERATURE: 0.1
    LOCAL_TEMPERATURE: 0.1
```

以上搜索半径均以 stride-8 feature pixel 为单位。实际范围应根据数据中的残余错位调整。

---

# 18. 建议代码目录

```text
maskdino_project/
├── maskdino/
│   ├── model.py
│   ├── transformer_decoder/
│   └── pixel_decoder/
│
├── sar_height/
│   ├── phase_descriptor.py
│   ├── sar_stem.py
│   ├── sar_resnet_fpn.py
│   ├── matching_projection.py
│   ├── global_correlation.py
│   ├── local_correlation.py
│   ├── geometry_decoder.py
│   ├── height_head.py
│   ├── height_loss.py
│   └── maskdino_sar_height.py
│
├── data/
│   ├── multimodal_mapper.py
│   └── sar_preprocessing.py
│
└── configs/
    └── maskdino_sar_height_baseline.yaml
```

各模块职责如下。

`phase_descriptor.py` 只负责从 `sinφ、cosφ、coherence` 生成 39 通道张量，不包含可学习参数。

`sar_stem.py` 实现 Phase Stem、Amplitude Stem、Amp Gate 和融合卷积。

`sar_resnet_fpn.py` 实现单个 ResNet18 `layer1-layer4` 和 FPN。

`matching_projection.py` 将 Optical stride-8 feature 和 SAR $P_3$ 投影到统一的 64 维 matching space。

`global_correlation.py` 输出每张图像的二维全局偏移。

`local_correlation.py` 根据 query、reference center 和 matching feature 输出每个 query 的二维局部偏移。

`geometry_decoder.py` 封装三层 Multi-Scale Deformable Cross-Attention，优先复用当前 MaskDINO 工程中的 MSDeformAttn。

`height_head.py` 从最终 query 预测高度。

`height_loss.py` 根据 MaskDINO matcher 的 indices 对齐预测高度和 GT 高度。

`maskdino_sar_height.py` 作为新的 meta architecture，组织 Optical、SAR、Alignment、Geometry Decoder 和 Height Head。

---

# 19. 需要修改的 MaskDINO 位置

现有 MaskDINO 源代码至少需要增加以下接口。

第一，在 transformer decoder forward 中返回最后一层 decoder query，而不是只返回分类、box 和 mask 预测。

第二，返回与最终 query 对应的 reference box。优先使用 decoder 内部 reference；如果当前实现没有保留，可以返回最终预测 box 的 sigmoid 结果。

第三，从 pixel decoder 返回一个 stride-8 feature。该 feature 只用于 matching，不改变原有 mask prediction。

第四，MaskDINO postprocessor 在选择最终实例时必须同时保留对应 query index。假设 postprocessor 最终选择的 query indices 为：

```python
selected_query_indices
```

则最终实例高度应通过：

```python
instance_heights = pred_heights[
    selected_query_indices
]
```

进行同步提取，确保每个 mask、box 和 height 来自同一个 query。

---

# 20. 训练与推理要求

训练时建议先加载已经可以正常工作的 MaskDINO checkpoint，并默认冻结 MaskDINO。新增 SAR Encoder、Matching Projection、Offset Module、Geometry Decoder 和 Height Head 从头训练。该设置不是复杂训练流程，而是为了先保证新增模块能够独立收敛，不破坏现有实例分割。

如果启用 synthetic global shift，应对原始四通道 SAR 施加已知二维平移，然后再生成 $X_{\text{phase}}$ 和 $X_{\text{amp}}$。全局偏移监督只要求 Global Correlation 恢复该平移。Local Correlation 和 Deformable Offset 由最终高度损失端到端优化。

推理时，MaskDINO 按原流程输出建筑实例。对于每一个最终保留的 query，附加其 `pred_height`。当前数据集 physical offset 为零，因此最终三维重建可直接将预测 mask 按预测高度垂直拉伸。

---

# 21. 开发验收标准

完成开发后，至少需要满足以下工程条件。

关闭所有 SAR 模块时，MaskDINO 的 class、box 和 mask 输出必须与原始代码一致。

SAR Encoder 对任意合法输入均应稳定输出：

```text
P2: [B, 256, H/4,  W/4]
P3: [B, 256, H/8,  W/8]
P4: [B, 256, H/16, W/16]
P5: [B, 256, H/32, W/32]
```

Global Offset 输出形状为：

```text
[B, 2]
```

Local Offset 输出形状为：

```text
[B, N_query, 2]
```

Geometry Decoder 最终 query 输出形状为：

```text
[B, N_query, 256]
```

Height Head 输出形状为：

```text
[B, N_query]
```

必须使用合成平移单元测试确认偏移符号。例如，当 SAR 相对于光学向右移动 8 个原始像素时，预测偏移应使 SAR sampling center 向右移动，而不是向左。

必须检查 `GLOBAL_ENABLED=False` 和 `LOCAL_CORR_ENABLED=False` 时，SAR reference box 与 Optical reference box 完全一致。

必须确认 Height Loss 只作用于 matcher 匹配到的正样本 query，不对 no-object query 计算高度损失。

必须确认 MaskDINO postprocessor 选择实例以后，mask、box、class 和 height 使用同一个 query index。

最终模型需要能够在一个很小的训练子集上过拟合，确认高度梯度能够依次传递到 Height Head、Geometry Decoder、Deformable Attention、SAR FPN、ResNet18 和两个 Stem。

---

# 22. Baseline 核心公式总结

SAR 特征提取：

$$
F_s
=
\text{FPN}
\left(
\text{ResNet18}
\left(
F_{\text{phase}}
+
G_{\text{amp}}\odot F_{\text{amp}}
\right)
\right)
$$

实例 SAR reference center：

$$
c_i^{sar}
=
c_i^{opt}
+
d_g
+
d_i^{local}
$$

Deformable Attention 采样位置：

$$
p_{i,h,l,k}^{t}
=
c_i^{opt}
+
d_g
+
d_i^{local}
+
d_{i,h,l,k}^{deform,t}
$$

三层 query 更新：

$$
Q_g^3
=
\text{GeometryDecoder}^{3}
\left(
Q_g^0,
F_s,
B_s
\right)
$$

最终高度：

$$
\hat H
=
\text{Softplus}
\left(
\text{MLP}_{height}
(Q_g^3)
\right)
$$

该 Baseline 的核心逻辑可以概括为：

> MaskDINO query 确定建筑实例；全局相关确定整幅 SAR 的共享错位；Local Correlation 确定每栋建筑的局部错位；Deformable Attention 在修正后的位置附近完成精细 SAR 特征采样；三层 decoder 将 SAR 信息写入 query；Height Head 从最终多模态 query 直接预测高度。
