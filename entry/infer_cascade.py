#!/usr/bin/env python
"""两阶段推理闭环：Focus -> 聚焦框 -> Refinement -> 回写全分辨率。

流程
----
1. 读取原始分辨率 CTA，按训练时的约定缩放至 (256, 256, 160)；
2. 用 Focus 模型预测低分辨率粗分割；
3. 对粗分割做阈值化 + 最大连通域 + 质心，映射回原始分辨率坐标，
   从原始分辨率图像中裁剪 128^3 聚焦框，并记录 origin / 原始 affine；
4. 用 Refinement 模型预测聚焦框；
5. 把细化结果按 origin 贴回零值全分辨率 volume，用原始 affine/spacing 保存。

网络选择
--------
``nnunetv2.utilities.get_network_from_plans`` 现在支持按阶段选择网络：
环境变量 ``MY_NNUNET_NETWORK=focus|refinement``、plans 中的自定义
``UNet_class_name``，以及 dataset 名里是否包含 ``refine``。
本脚本还会读取 checkpoint 的 state_dict 自动判断应该用哪个网络，
因此已有的粗/细模型都不需要重新训练。

示例
----
python entry/infer_cascade.py \\
    -i /path/to/imagesTs_fullres \\
    -o /path/to/cascade_output \\
    --focus_model  /path/to/Dataset505_WMC_only_foucs/nnUNetTrainer__nnUNetPlans__3d_lowres \\
    --refine_model /path/to/Dataset508_WMC_focus_as_refine_train/nnUNetTrainer__nnUNetPlans__3d_fullres \\
    --focus_fold all --refine_fold all \\
    --focus_checkpoint checkpoint_zyh_best.pth \\
    --refine_checkpoint checkpoint_best.pth \\
    --device cuda
"""

from __future__ import annotations

import argparse
import glob
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.focus_cube import (  # noqa: E402
    CUBE_SIZE,
    DEFAULT_LOWRES_SHAPE,
    as_3d,
    extract_cube,
    largest_component_centroid,
    paste_cube,
    resample_to_shape,
    scale_center,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Target-aware coarse-to-fine inference (focus -> cube -> refinement -> full res)."
    )
    parser.add_argument("-i", "--input_dir", required=True,
                        help="原始分辨率输入目录，文件名需符合 nnU-Net 约定：<case>_0000.nii.gz")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="最终全分辨率预测输出目录")

    parser.add_argument("--focus_model", default=None,
                        help="Focus 模型目录（包含 plans.json/dataset.json/fold_*），"
                             "提供 --coarse_pred_dir 时可省略")
    parser.add_argument("--refine_model", default=None,
                        help="Refinement 模型目录（包含 plans.json/dataset.json/fold_*），"
                             "提供 --refined_dir 时可省略")
    parser.add_argument("--focus_fold", default="all", help="Focus 模型 fold，默认 all")
    parser.add_argument("--refine_fold", default="all", help="Refinement 模型 fold，默认 all")
    parser.add_argument("--focus_checkpoint", default=None,
                        help="Focus checkpoint 文件名；默认自动在 fold 目录中寻找 best/final/zyh_best/latest")
    parser.add_argument("--refine_checkpoint", default=None,
                        help="Refinement checkpoint 文件名；默认同上")
    parser.add_argument("--focus_network", default="auto", choices=("auto", "focus", "refinement"),
                        help="Focus 阶段的网络类型；auto 时从 checkpoint 自动判断")
    parser.add_argument("--refine_network", default="auto", choices=("auto", "focus", "refinement"),
                        help="Refinement 阶段的网络类型；auto 时从 checkpoint 自动判断")

    parser.add_argument("--lowres_dir", default=None,
                        help="复用已生成的低分辨率图像目录；不提供则自动缩放")
    parser.add_argument("--coarse_pred_dir", default=None,
                        help="复用已有的 Focus 预测目录（<case>.nii.gz）；提供后跳过 Focus 推理")
    parser.add_argument("--refined_dir", default=None,
                        help="复用已有的 Refinement 预测目录（<case>.nii.gz）；提供后跳过 Refinement 推理")

    parser.add_argument("--downsample_shape", default=",".join(str(i) for i in DEFAULT_LOWRES_SHAPE),
                        help="Focus 输入尺寸，默认 256,256,160（与训练脚本一致）")
    parser.add_argument("--cube_size", type=int, default=CUBE_SIZE, help="聚焦框边长，默认 128")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="粗分割二值化阈值，默认 0.10；nnU-Net 导出的 label map 取 >=0.10 即前景")
    parser.add_argument("--device", default="cuda", help="cuda / cpu / mps")
    parser.add_argument("--npp", type=int, default=1, help="预处理进程数，Windows 下建议 1")
    parser.add_argument("--nps", type=int, default=1, help="导出进程数，Windows 下建议 1")
    parser.add_argument("--cleanup", action="store_true", help="完成后删除 _cascade_tmp 中间目录")
    return parser.parse_args()


def parse_shape(text: str):
    values = tuple(int(i) for i in text.replace(" ", "").split(",") if i)
    if len(values) != 3:
        raise ValueError(f"--downsample_shape must be 3 comma-separated integers, got {text!r}")
    return values


def list_cases(input_dir: str):
    """返回 {case_id: path}。支持 <case>_0000.nii.gz / <case>_0000.nii。"""
    cases = {}
    for ext in (".nii.gz", ".nii"):
        for path in glob.glob(os.path.join(input_dir, "*_0000" + ext)):
            case = os.path.basename(path)[: -len("_0000" + ext)]
            cases[case] = path
    if not cases:
        raise FileNotFoundError(
            f"no '<case>_0000.nii.gz' files found in {input_dir!r}. "
            "Please provide an nnU-Net-style input folder."
        )
    return dict(sorted(cases.items()))


def resolve_checkpoint(model_dir: str, fold: str, requested: Optional[str]):
    fold_dir = os.path.join(model_dir, f"fold_{fold}")
    if not os.path.isdir(fold_dir):
        raise FileNotFoundError(f"fold directory not found: {fold_dir}")
    if requested:
        if os.path.isfile(os.path.join(fold_dir, requested)):
            return requested
        raise FileNotFoundError(f"requested checkpoint not found: {os.path.join(fold_dir, requested)}")

    candidates = [
        "checkpoint_zyh_best.pth",
        "checkpoint_best.pth",
        "checkpoint_final.pth",
        "checkpoint_latest.pth",
    ]
    for name in candidates:
        if os.path.isfile(os.path.join(fold_dir, name)):
            return name
    raise FileNotFoundError(
        f"no checkpoint found in {fold_dir}; tried {candidates}. "
        "Use --focus_checkpoint / --refine_checkpoint to specify one."
    )


def detect_network_from_checkpoint(model_dir: str, fold: str, checkpoint_name: str):
    """DTS 权重（transformer_skip / swin_layer）是否存在，决定用 refinement 还是 focus。"""
    path = os.path.join(model_dir, f"fold_{fold}", checkpoint_name)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = checkpoint.get("network_weights", checkpoint)
    for key in weights.keys():
        if "transformer_skip" in key or "swin_layer" in key:
            return "refinement"
    return "focus"


def run_predictor(model_dir: str, input_dir: str, output_dir: str, fold: str,
                  checkpoint_name: str, network_variant: str, device: torch.device,
                  npp: int, nps: int):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    os.makedirs(output_dir, exist_ok=True)
    previous = os.environ.get("MY_NNUNET_NETWORK")
    os.environ["MY_NNUNET_NETWORK"] = network_variant
    predictor = None
    try:
        print(f"[cascade] running {network_variant} predictor: {model_dir} -> {output_dir}")
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_gpu=(device.type == "cuda"),
            device=device,
            verbose=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            model_dir, use_folds=(fold,), checkpoint_name=checkpoint_name
        )
        predictor.predict_from_files(
            input_dir,
            output_dir,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=npp,
            num_processes_segmentation_export=nps,
        )
    finally:
        if previous is None:
            os.environ.pop("MY_NNUNET_NETWORK", None)
        else:
            os.environ["MY_NNUNET_NETWORK"] = previous
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_fullres_metas(cases):
    """只读取原始分辨率图像的几何信息，不做下采样。"""
    metas = {}
    for case, path in cases.items():
        image = nib.load(path)
        data = as_3d(image.get_fdata(dtype=np.float32))
        metas[case] = {
            "input_path": path,
            "full_shape": [int(i) for i in data.shape],
            "full_affine": image.affine.tolist(),
        }
    return metas


def prepare_lowres(cases, lowres_dir: str, target_shape):
    """按训练约定生成低分辨率输入，并返回每个病例的原始几何信息。"""
    os.makedirs(lowres_dir, exist_ok=True)
    metas = build_fullres_metas(cases)
    for case, path in cases.items():
        image = nib.load(path)
        data = as_3d(image.get_fdata(dtype=np.float32))
        low = resample_to_shape(data, target_shape)
        out_path = os.path.join(lowres_dir, f"{case}_0000.nii.gz")
        # 训练脚本 utils/focus_trainset_generator.get_downsample 也是用 eye(4) 保存低分辨率图像
        nib.save(nib.Nifti1Image(low.astype(np.float32), np.eye(4)), out_path)
        metas[case]["lowres_path"] = out_path
        metas[case]["lowres_shape"] = [int(i) for i in low.shape]
        print(f"[cascade] downsample {case}: {tuple(data.shape)} -> {tuple(low.shape)}")
    return metas


def build_cubes(metas, coarse_pred_dir: str, cubes_dir: str, cube_size: int,
                threshold: float, metadata_path: str):
    """从粗分割中定位最大连通域，裁剪原始分辨率的聚焦框并记录 origin。"""
    os.makedirs(cubes_dir, exist_ok=True)
    records = {}
    for case, meta in metas.items():
        pred_path = os.path.join(coarse_pred_dir, f"{case}.nii.gz")
        record = dict(meta)
        record["has_cube"] = False

        if not os.path.isfile(pred_path):
            print(f"[cascade][warn] no coarse prediction for {case}: {pred_path}")
            records[case] = record
            continue

        pred = as_3d(nib.load(pred_path).get_fdata(dtype=np.float32))
        centroid_low = largest_component_centroid(pred >= threshold)
        if centroid_low is None:
            print(f"[cascade][warn] empty coarse prediction for {case}; final output will be empty")
            records[case] = record
            continue

        full_image = nib.load(meta["input_path"])
        full = as_3d(full_image.get_fdata(dtype=np.float32))
        centroid_full = scale_center(centroid_low, pred.shape, full.shape)
        cube, origin = extract_cube(full, centroid_full, cube_size=cube_size)

        cube_path = os.path.join(cubes_dir, f"{case}_0000.nii.gz")
        # 训练时的聚焦框也是用 eye(4) 保存的，保持一致
        nib.save(nib.Nifti1Image(cube.astype(np.float32), np.eye(4)), cube_path)

        record.update({
            "has_cube": True,
            "cube_path": cube_path,
            "origin": [int(i) for i in origin],
            "pred_shape": [int(i) for i in pred.shape],
            "centroid_lowres": [float(i) for i in centroid_low],
            "centroid_fullres": [float(i) for i in centroid_full],
        })
        records[case] = record
        print(f"[cascade] cube {case}: origin={origin.tolist()}, "
              f"lowres_centroid={np.round(centroid_low, 1).tolist()}, "
              f"fullres_centroid={np.round(centroid_full, 1).tolist()}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return records


def assemble_fullres(records, refined_dir: str, output_dir: str):
    """把细化后的 cube 按 origin 贴回全分辨率 volume，保留原始 affine。"""
    os.makedirs(output_dir, exist_ok=True)
    for case, record in records.items():
        full_image = nib.load(record["input_path"])
        full_shape = tuple(int(i) for i in record["full_shape"])

        if record.get("has_cube"):
            refined_path = os.path.join(refined_dir, f"{case}.nii.gz")
        else:
            refined_path = None

        if refined_path is None or not os.path.isfile(refined_path):
            if record.get("has_cube"):
                print(f"[cascade][warn] no refined prediction for {case}: {refined_path}")
            final = np.zeros(full_shape, dtype=np.uint8)
        else:
            cube_pred = as_3d(nib.load(refined_path).get_fdata(dtype=np.float32))
            cube_pred = (cube_pred > 0).astype(np.uint8)
            final = paste_cube(cube_pred, record["origin"], full_shape)

        header = full_image.header.copy()
        header.set_data_dtype(np.uint8)
        out_image = nib.Nifti1Image(final, full_image.affine, header=header)
        out_path = os.path.join(output_dir, f"{case}.nii.gz")
        nib.save(out_image, out_path)
        print(f"[cascade] wrote {out_path}  (foreground voxels={int(final.sum())})")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tmp_root = os.path.join(args.output_dir, "_cascade_tmp")
    os.makedirs(tmp_root, exist_ok=True)

    cases = list_cases(args.input_dir)
    print(f"[cascade] {len(cases)} case(s) found")

    # ---------- 1. 低分辨率输入 ----------
    # 如果已经有粗分割结果，就不再浪费时间做下采样（Focus 推理仍需要 lowres_dir）
    if args.coarse_pred_dir:
        metas = build_fullres_metas(cases)
        lowres_dir = args.lowres_dir or os.path.join(tmp_root, "lowres")
    elif args.lowres_dir:
        lowres_dir = args.lowres_dir
        metas = build_fullres_metas(cases)
        for case in cases:
            low_path = os.path.join(lowres_dir, f"{case}_0000.nii.gz")
            if not os.path.isfile(low_path):
                raise FileNotFoundError(f"missing lowres input for {case}: {low_path}")
            metas[case]["lowres_path"] = low_path
            metas[case]["lowres_shape"] = [int(i) for i in nib.load(low_path).shape]
    else:
        lowres_dir = os.path.join(tmp_root, "lowres")
        metas = prepare_lowres(cases, lowres_dir, parse_shape(args.downsample_shape))

    # ---------- 2. Focus 预测 ----------
    if args.coarse_pred_dir:
        coarse_pred_dir = args.coarse_pred_dir
        print(f"[cascade] reuse coarse predictions: {coarse_pred_dir}")
    else:
        if not args.focus_model:
            raise ValueError("--focus_model is required unless --coarse_pred_dir is provided")
        coarse_pred_dir = os.path.join(tmp_root, "coarse")
        checkpoint = resolve_checkpoint(args.focus_model, args.focus_fold, args.focus_checkpoint)
        variant = args.focus_network
        if variant == "auto":
            variant = detect_network_from_checkpoint(args.focus_model, args.focus_fold, checkpoint)
        print(f"[cascade] focus checkpoint={checkpoint}, network={variant}")
        run_predictor(
            args.focus_model, lowres_dir, coarse_pred_dir, args.focus_fold,
            checkpoint, variant, torch.device(args.device), args.npp, args.nps,
        )

    # ---------- 3. 聚焦框 ----------
    cubes_dir = os.path.join(tmp_root, "cubes")
    metadata_path = os.path.join(args.output_dir, "cascade_metadata.json")
    records = build_cubes(metas, coarse_pred_dir, cubes_dir, args.cube_size,
                          args.threshold, metadata_path)

    # ---------- 4. Refinement 预测 ----------
    if args.refined_dir:
        refined_dir = args.refined_dir
        print(f"[cascade] reuse refined predictions: {refined_dir}")
    else:
        if not args.refine_model:
            raise ValueError("--refine_model is required unless --refined_dir is provided")
        refined_dir = os.path.join(tmp_root, "refined")
        checkpoint = resolve_checkpoint(args.refine_model, args.refine_fold, args.refine_checkpoint)
        variant = args.refine_network
        if variant == "auto":
            variant = detect_network_from_checkpoint(args.refine_model, args.refine_fold, checkpoint)
        print(f"[cascade] refinement checkpoint={checkpoint}, network={variant}")
        run_predictor(
            args.refine_model, cubes_dir, refined_dir, args.refine_fold,
            checkpoint, variant, torch.device(args.device), args.npp, args.nps,
        )

    # ---------- 5. 回写全分辨率 ----------
    assemble_fullres(records, refined_dir, args.output_dir)
    print(f"[cascade] done. metadata: {metadata_path}")

    if args.cleanup:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"[cascade] removed temporary directory: {tmp_root}")


if __name__ == "__main__":
    main()
