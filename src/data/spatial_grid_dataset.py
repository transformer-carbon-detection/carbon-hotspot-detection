from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


REQUIRED_ARRAYS = ["stream_a", "stream_b", "label_cls", "label_reg", "mask"]


def load_tensor_index(tensor_index_csv: str | Path) -> list[str]:
    """Read a tensor index CSV and return tensor paths in row order."""
    index_path = Path(tensor_index_csv)
    if not index_path.exists():
        raise FileNotFoundError(f"Tensor index CSV not found: {index_path}")

    df = pd.read_csv(index_path)
    if "tensor_path" not in df.columns:
        raise ValueError(
            f"Tensor index CSV must contain 'tensor_path'. "
            f"Available columns: {list(df.columns)}"
        )

    return df["tensor_path"].astype(str).tolist()


class SpatialGridDataset(Dataset):
    """Patch-based dataset for MS-MCA spatial tensor .npz files."""

    def __init__(
        self,
        npz_paths: list[str | Path] | None = None,
        tensor_index_csv: str | Path | None = None,
        crop_size: int = 16,
        target: str = "center",
        use_regression: bool = True,
        transform: Any = None,
        seed: int = 42,
    ) -> None:
        if crop_size <= 0:
            raise ValueError(f"crop_size must be positive, got: {crop_size}")
        if target != "center":
            raise ValueError("Only target='center' is currently supported.")
        if npz_paths is None and tensor_index_csv is None:
            raise ValueError("Provide either npz_paths or tensor_index_csv.")
        if npz_paths is not None and tensor_index_csv is not None:
            raise ValueError("Provide only one of npz_paths or tensor_index_csv.")

        self.crop_size = crop_size
        self.target = target
        self.use_regression = use_regression
        self.transform = transform
        self.seed = seed

        path_values = npz_paths if npz_paths is not None else load_tensor_index(tensor_index_csv)
        self.npz_paths = [Path(path) for path in path_values]
        self.samples: list[tuple[int, int, int]] = []
        self.monthly_tensors: list[dict[str, Any]] = []

        for file_idx, npz_path in enumerate(self.npz_paths):
            tensor_dict = self._load_npz(npz_path)
            self.monthly_tensors.append(tensor_dict)

            valid_rows, valid_cols = np.where(tensor_dict["mask"] == 1)
            for row, col in zip(valid_rows.tolist(), valid_cols.tolist(), strict=True):
                self.samples.append((file_idx, row, col))

        if not self.samples:
            raise ValueError("No valid mask cells found in the provided tensor files.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_idx, row, col = self.samples[idx]
        tensor_dict = self.monthly_tensors[file_idx]

        stream_a_patch = self._crop_with_padding(tensor_dict["stream_a"], row, col)
        stream_b_patch = self._crop_with_padding(tensor_dict["stream_b"], row, col)
        mask_patch = self._crop_with_padding(tensor_dict["mask"], row, col)

        sample = {
            "stream_a": torch.as_tensor(stream_a_patch, dtype=torch.float32),
            "stream_b": torch.as_tensor(stream_b_patch, dtype=torch.float32),
            "label_cls": torch.as_tensor(
                tensor_dict["label_cls"][row, col],
                dtype=torch.long,
            ),
            "label_reg": torch.as_tensor(
                tensor_dict["label_reg"][row, col],
                dtype=torch.float32,
            ),
            "mask": torch.as_tensor(mask_patch, dtype=torch.float32),
            "date": tensor_dict["date"],
            "row": row,
            "col": col,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def print_debug_summary(self) -> None:
        label_values = []
        for tensor_dict in self.monthly_tensors:
            valid_mask = tensor_dict["mask"] == 1
            label_values.extend(tensor_dict["label_cls"][valid_mask].tolist())

        first_tensor = self.monthly_tensors[0]
        first_sample = self[0]

        print("SpatialGridDataset summary")
        print(f"NPZ files loaded: {len(self.monthly_tensors)}")
        print(f"Total valid samples: {len(self)}")
        print(f"stream_a shape: {first_tensor['stream_a'].shape}")
        print(f"stream_b shape: {first_tensor['stream_b'].shape}")
        print("label_cls distribution:")
        print(pd.Series(label_values).value_counts().sort_index())
        print("First sample shapes:")
        print(f"stream_a: {tuple(first_sample['stream_a'].shape)}")
        print(f"stream_b: {tuple(first_sample['stream_b'].shape)}")
        print(f"mask: {tuple(first_sample['mask'].shape)}")
        print(f"label_cls: {tuple(first_sample['label_cls'].shape)}")
        print(f"label_reg: {tuple(first_sample['label_reg'].shape)}")

    def _load_npz(self, npz_path: Path) -> dict[str, Any]:
        if not npz_path.exists():
            raise FileNotFoundError(f"Tensor .npz file not found: {npz_path}")

        with np.load(npz_path, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_ARRAYS if key not in data.files]
            if missing:
                raise ValueError(f"{npz_path} missing required arrays: {missing}")

            tensor_dict = {
                "stream_a": data["stream_a"].astype(np.float32),
                "stream_b": data["stream_b"].astype(np.float32),
                "label_cls": data["label_cls"].astype(np.int64),
                "label_reg": data["label_reg"].astype(np.float32),
                "mask": data["mask"].astype(np.float32),
                "date": _read_npz_date(data, npz_path),
            }

        self._validate_tensor_shapes(npz_path, tensor_dict)
        return tensor_dict

    def _validate_tensor_shapes(self, npz_path: Path, tensor_dict: dict[str, Any]) -> None:
        stream_a = tensor_dict["stream_a"]
        stream_b = tensor_dict["stream_b"]
        label_cls = tensor_dict["label_cls"]
        label_reg = tensor_dict["label_reg"]
        mask = tensor_dict["mask"]

        if stream_a.ndim != 3 or stream_a.shape[0] != 3:
            raise ValueError(
                f"{npz_path} stream_a must have shape [3, H, W], "
                f"got {stream_a.shape}"
            )
        if stream_b.ndim != 3 or stream_b.shape[0] != 4:
            raise ValueError(
                f"{npz_path} stream_b must have shape [5, H, W], "
                f"got {stream_b.shape}"
            )

        spatial_shape = stream_a.shape[1:]
        if stream_b.shape[1:] != spatial_shape:
            raise ValueError(
                f"{npz_path} stream_a and stream_b spatial shapes differ: "
                f"{spatial_shape} vs {stream_b.shape[1:]}"
            )

        for name, array in [
            ("label_cls", label_cls),
            ("label_reg", label_reg),
            ("mask", mask),
        ]:
            if array.shape != spatial_shape:
                raise ValueError(
                    f"{npz_path} {name} shape must match H/W {spatial_shape}, "
                    f"got {array.shape}"
                )

        if not np.any(mask == 1):
            raise ValueError(f"{npz_path} contains no valid mask cells.")

    def _crop_with_padding(self, array: np.ndarray, row: int, col: int) -> np.ndarray:
        pad_before = self.crop_size // 2
        pad_after = self.crop_size - pad_before - 1

        row_start = row - pad_before
        row_end = row + pad_after + 1
        col_start = col - pad_before
        col_end = col + pad_after + 1

        if array.ndim == 3:
            _, height, width = array.shape
            output = np.zeros(
                (array.shape[0], self.crop_size, self.crop_size),
                dtype=array.dtype,
            )
            src_row_start = max(row_start, 0)
            src_row_end = min(row_end, height)
            src_col_start = max(col_start, 0)
            src_col_end = min(col_end, width)
            dst_row_start = src_row_start - row_start
            dst_col_start = src_col_start - col_start
            output[
                :,
                dst_row_start : dst_row_start + (src_row_end - src_row_start),
                dst_col_start : dst_col_start + (src_col_end - src_col_start),
            ] = array[:, src_row_start:src_row_end, src_col_start:src_col_end]
            return output

        if array.ndim == 2:
            height, width = array.shape
            output = np.zeros((self.crop_size, self.crop_size), dtype=array.dtype)
            src_row_start = max(row_start, 0)
            src_row_end = min(row_end, height)
            src_col_start = max(col_start, 0)
            src_col_end = min(col_end, width)
            dst_row_start = src_row_start - row_start
            dst_col_start = src_col_start - col_start
            output[
                dst_row_start : dst_row_start + (src_row_end - src_row_start),
                dst_col_start : dst_col_start + (src_col_end - src_col_start),
            ] = array[src_row_start:src_row_end, src_col_start:src_col_end]
            return output

        raise ValueError(f"Expected 2D or 3D array, got shape {array.shape}")


def _read_npz_date(data: np.lib.npyio.NpzFile, npz_path: Path) -> str:
    if "date" not in data.files:
        return npz_path.stem

    raw_date = data["date"]
    if raw_date.shape == ():
        return str(raw_date.item())
    return str(raw_date.tolist())


def debug_spatial_grid_dataset(
    npz_paths: list[str | Path] | None = None,
    tensor_index_csv: str | Path | None = None,
    crop_size: int = 16,
) -> SpatialGridDataset:
    dataset = SpatialGridDataset(
        npz_paths=npz_paths,
        tensor_index_csv=tensor_index_csv,
        crop_size=crop_size,
    )
    dataset.print_debug_summary()
    return dataset
