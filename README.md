# 网络整体工作流程

## 网络训练
### 聚焦阶段
1. 全部将样本进行下采样——utils.focus_trainset_generator.py
   1. 下采样的尺寸为[256, 256, 160]
2. 对网络进行训练（默认使用 `my_model/coarse_net.py` 的普通 U-Net）

### 精炼阶段
1. 训练数据由原始数据自动生成-utils.refinement_trainset_generator.py
   1. 生成的尺寸为[128, 128, 128]
2. 对网络进行训练（使用 `my_model/refinement_net.py`，含 DTS-H / DTS-V）

## 阶段与网络的选择

`nnunetv2/utilities/get_network_from_plans.py` 现在会按阶段选择网络，优先级：

1. 环境变量 `MY_NNUNET_NETWORK`：`focus` / `refinement`；
2. plans 中的 `UNet_class_name`：`PlainConvUNetFocus` / `PlainConvUNetRefinement`；
3. dataset 名里包含 `refine` 时自动使用 Refinement 网络，否则使用 Focus 网络。

例如，`Dataset508_WMC_focus_as_refine_train` 会自动使用 Refinement 网络，
`Dataset505_WMC_only_foucs` 会自动使用 Focus 网络。若命名不包含 `refine`，
训练/推理前显式设置环境变量即可：

```bash
MY_NNUNET_NETWORK=refinement nnUNetv2_train 508 3d_fullres all
MY_NNUNET_NETWORK=focus      nnUNetv2_train 505 3d_lowres all
```

已有的粗/细模型不需要重新训练：网络结构类没有变，只是从“注释切换”改成按阶段选择。

## 测试阶段（完整闭环）

新增 `entry/infer_cascade.py`，一条命令完成：

1. 原始分辨率 CTA 按训练约定缩放到 [256, 256, 160]；
2. Focus 网络预测低分辨率粗分割；
3. 0.10 阈值 + 最大连通域 + 质心，映射回原始分辨率，裁剪 [128, 128, 128] 聚焦框，
   同时记录 `origin`、原始 shape 和 affine；
4. Refinement 网络预测聚焦框；
5. 按 `origin` 把细化结果贴回零值全分辨率 volume，用原始 affine/spacing 保存。

```bash
python entry/infer_cascade.py \
    -i /path/to/imagesTs_fullres \
    -o /path/to/cascade_output \
    --focus_model  /path/to/Dataset505_WMC_only_foucs/nnUNetTrainer__nnUNetPlans__3d_lowres \
    --refine_model /path/to/Dataset508_WMC_focus_as_refine_train/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --focus_fold all --refine_fold all \
    --focus_checkpoint checkpoint_zyh_best.pth \
    --refine_checkpoint checkpoint_best.pth \
    --device cuda
```

输出：

- `output_dir/<case>.nii.gz`：最终全分辨率预测；
- `output_dir/cascade_metadata.json`：每个病例的 cube origin / 原始 shape / affine；
- `output_dir/_cascade_tmp/`：下采样图像、粗分割、聚焦框、细化结果，便于排查
  （`--cleanup` 可在完成后删除）。

也可以复用中间结果：`--lowres_dir`、`--coarse_pred_dir`、`--refined_dir`，
分别跳过下采样、Focus 推理、Refinement 推理。网络类型默认从 checkpoint 的
state_dict 自动判断（含 `transformer_skip` / `swin_layer` 即为 Refinement 网络），
不需要手工改 import。


