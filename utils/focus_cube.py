"""聚焦框（focus cube）相关的纯工具函数。

这里刻意只依赖 numpy / scipy / nibabel / torch，不依赖 nnU-Net 的推理器，
方便单独测试，也方便在 ``entry/infer_cascade.py`` 里复用。

坐标约定
--------
所有数组都按 NIfTI 的 ``(X, Y, Z)`` 顺序处理。
训练阶段 ``utils/focus_trainset_generator.get_downsample`` 把原图用
nearest 插值缩放到固定的 ``(256, 256, 160)``，并以 ``np.eye(4)`` 保存，
因此推理阶段也沿用“索引空间按比例缩放”的约定，而不是物理空间仿射变换。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

CUBE_SIZE = 128
DEFAULT_LOWRES_SHAPE = (256, 256, 160)


def as_3d(data: np.ndarray) -> np.ndarray:
    """把带多余尾维（例如 channel 维）的数组压成 3D。"""
    data = np.asarray(data)
    while data.ndim > 3:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"expected a 3D array, got shape {data.shape}")
    return data


def resample_to_shape(data: np.ndarray, target_shape) -> np.ndarray:
    """按索引空间做 nearest 缩放，和训练时的 ``F.interpolate`` 保持一致。"""
    import torch
    import torch.nn.functional as F

    data = as_3d(data)
    target_shape = tuple(int(i) for i in target_shape)
    tensor = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
    tensor = tensor[None, None]
    tensor = F.interpolate(tensor, size=target_shape, mode="nearest")
    return tensor[0, 0].cpu().numpy()


def largest_component_centroid(mask: np.ndarray):
    """返回最大连通域（6 邻接，scipy 默认结构）的质心；没有前景返回 None。"""
    mask = as_3d(mask) > 0
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return None
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = int(sizes.argmax())
    if sizes[largest_label] == 0:
        return None
    centroid = ndimage.center_of_mass(mask, labeled, largest_label)
    return np.asarray(centroid, dtype=np.float64)


def extract_cube(data: np.ndarray, center, cube_size: int = CUBE_SIZE):
    """以 ``center`` 为中心裁剪 ``cube_size^3``。

    返回 ``(cube, origin)``。如果图像某一边小于 cube_size，对应方向补零；
    否则把 origin 夹到 ``[0, dim - cube_size]``，与论文“必要时平移聚焦框”一致。
    """
    data = as_3d(data)
    shape = np.asarray(data.shape, dtype=np.int64)
    center = np.asarray(center, dtype=np.float64)
    origin = np.round(center - cube_size / 2.0).astype(np.int64)
    origin = np.clip(origin, 0, np.maximum(shape - cube_size, 0))

    cube = np.zeros((cube_size, cube_size, cube_size), dtype=data.dtype)
    src_stop = np.minimum(origin + cube_size, shape)
    dst_stop = src_stop - origin
    cube[: dst_stop[0], : dst_stop[1], : dst_stop[2]] = data[
        origin[0] : src_stop[0],
        origin[1] : src_stop[1],
        origin[2] : src_stop[2],
    ]
    return cube, origin


def paste_cube(cube: np.ndarray, origin, full_shape) -> np.ndarray:
    """把 cube 贴回 ``full_shape`` 的零值 volume，返回新的数组。"""
    cube = as_3d(cube)
    full_shape = np.asarray(full_shape, dtype=np.int64)
    origin = np.asarray(origin, dtype=np.int64)
    out = np.zeros(tuple(int(i) for i in full_shape), dtype=cube.dtype)

    # origin 可能因为图像小于 cube 而为 0，这里统一做一次安全裁剪
    origin = np.clip(origin, 0, np.maximum(full_shape - 1, 0))
    src_stop = np.minimum(np.asarray(cube.shape, dtype=np.int64) + origin, full_shape)
    dst_stop = src_stop - origin
    if np.any(dst_stop <= 0):
        return out
    out[origin[0] : src_stop[0], origin[1] : src_stop[1], origin[2] : src_stop[2]] = cube[
        : dst_stop[0], : dst_stop[1], : dst_stop[2]
    ]
    return out


def scale_center(center_lowres, lowres_shape, full_shape):
    """把低分辨率索引坐标按各轴比例映射到原始分辨率索引坐标。"""
    center_lowres = np.asarray(center_lowres, dtype=np.float64)
    lowres_shape = np.asarray(lowres_shape, dtype=np.float64)
    full_shape = np.asarray(full_shape, dtype=np.float64)
    if np.any(lowres_shape <= 0):
        raise ValueError(f"invalid lowres shape: {lowres_shape}")
    return center_lowres * (full_shape / lowres_shape)


if __name__ == "__main__":
    # 无依赖自测：裁剪后贴回应还原中心区域
    vol = np.arange(40 * 50 * 60, dtype=np.float32).reshape(40, 50, 60)
    cube, origin = extract_cube(vol, (20, 25, 30), cube_size=16)
    rec = paste_cube(cube, origin, vol.shape)
    assert np.allclose(rec[origin[0] : origin[0] + 16,
                           origin[1] : origin[1] + 16,
                           origin[2] : origin[2] + 16], cube)
    print("focus_cube self-test passed, origin =", origin.tolist())
