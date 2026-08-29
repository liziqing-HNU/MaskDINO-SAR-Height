# SP6 Baseline

原有的光学-only 高度回归说明已由完整的 SAR 弱对齐 Baseline 取代。

请阅读 [SP6_SAR_HEIGHT.md](SP6_SAR_HEIGHT.md)。主配置从随机权重联合训练 MaskDINO 与 SAR 分支：

```text
configs/sp6/instance-segmentation/maskdino_R50_scratch_bs2_acc8_100ep_height.yaml
```
