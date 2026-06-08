"""Point-cloud datasets: synthetic primitives + a file-folder loader.

Sample protocol: ``(points (N, 3) float, target)`` where the target is an int
class (classification) or per-point ``(N,)`` int64 labels (segmentation).
Deterministic per (split, index), offline, CPU-trainable in minutes — the
3D analogue of the synthetic shapes fixtures.

File loaders cover ``.npy/.npz`` (numpy), ``.ply`` (``plyfile``, in the
``[3d]`` extra), ``.off``, and ``.xyz/.pts/.txt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from image_analytics.core.registry import DATASETS

_SPLIT_OFFSETS = {"train": 0, "val": 1, "test": 2}

PRIMITIVE_CLASSES = ("cube", "sphere", "plane", "cylinder")
NUM_SEG_PARTS = 4


# ---------------------------------------------------------------------------
# Surface sampling for each primitive (unit-scale, pre-pose)
# ---------------------------------------------------------------------------


def _sample_primitive(label: int, n: int, gen: torch.Generator) -> torch.Tensor:
    if label == 0:  # cube surface
        pts = torch.rand(n, 3, generator=gen) * 2 - 1
        faces = torch.randint(0, 6, (n,), generator=gen)
        axis, side = faces // 2, (faces % 2) * 2 - 1
        pts[torch.arange(n), axis] = side.float()
        return pts
    if label == 1:  # sphere surface
        pts = torch.randn(n, 3, generator=gen)
        return pts / pts.norm(dim=1, keepdim=True).clamp(min=1e-6)
    if label == 2:  # plane (z ~ 0)
        pts = torch.rand(n, 3, generator=gen) * 2 - 1
        pts[:, 2] = 0.0
        return pts
    # cylinder: side wall + caps
    theta = torch.rand(n, generator=gen) * 2 * torch.pi
    z = torch.rand(n, generator=gen) * 2 - 1
    pts = torch.stack([torch.cos(theta), torch.sin(theta), z], dim=1)
    cap = torch.rand(n, generator=gen) < 0.2
    r = torch.rand(int(cap.sum()), generator=gen).sqrt()
    pts[cap, 0] *= r
    pts[cap, 1] *= r
    pts[cap, 2] = torch.sign(pts[cap, 2])
    return pts


def _seg_labels(points: torch.Tensor) -> torch.Tensor:
    """Per-point part labels by angular sector about z (learnable from xy)."""
    angle = torch.atan2(points[:, 1], points[:, 0]) + torch.pi
    return (angle / (2 * torch.pi) * NUM_SEG_PARTS).long().clamp(max=NUM_SEG_PARTS - 1)


@DATASETS.register("synthetic_pointcloud")
class SyntheticPointCloud(Dataset):
    """Points sampled from primitive surfaces with random pose + jitter.

    ``task="classification"`` -> int shape label; ``task="segmentation"`` ->
    per-point sector labels.
    """

    CLASSES = PRIMITIVE_CLASSES

    def __init__(
        self,
        root: str | None = None,  # unused; registry protocol compatibility
        split: str = "train",
        transform: Callable | None = None,
        size: int = 256,
        num_points: int = 1024,
        task: str = "classification",
    ) -> None:
        if task not in ("classification", "segmentation"):
            raise ValueError(f"task must be classification|segmentation, got {task!r}")
        self.split = split
        self.transform = transform
        self.size = size
        self.num_points = num_points
        self.task = task
        self._offset = _SPLIT_OFFSETS.get(split, 3) * 1_000_003

    @property
    def num_classes(self) -> int:
        return NUM_SEG_PARTS if self.task == "segmentation" else len(self.CLASSES)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        gen = torch.Generator().manual_seed(self._offset + index)
        label = int(torch.randint(0, len(self.CLASSES), (1,), generator=gen))
        points = _sample_primitive(label, self.num_points, gen)

        # Random upright pose: z-rotation + small jitter (pre-transform).
        theta = torch.rand(1, generator=gen) * 2 * torch.pi
        c, s = torch.cos(theta), torch.sin(theta)
        rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).reshape(3, 3)
        points = points @ rot.t()
        points = points + torch.randn(self.num_points, 3, generator=gen) * 0.01

        if self.transform is not None:
            points = self.transform(points)
        # Segmentation labels are derived from the *final* points so they stay
        # consistent with what the model sees (normalization/rotation included).
        target = _seg_labels(points) if self.task == "segmentation" else label
        return points, target


@DATASETS.register("synthetic_pointcloud_det")
class SyntheticPointCloud3DDetection(Dataset):
    """A ground plane with cuboid point clusters and their 3D boxes.

    Sample: ``(points (N, 3), {"boxes_3d": (M, 7), "labels": (M,)})`` with
    axis-aligned boxes ``(x, y, z, dx, dy, dz, yaw=0)``. Single foreground
    class. Deterministic per (split, index).
    """

    CLASSES = ("object",)

    def __init__(
        self,
        root: str | None = None,  # unused; registry protocol compatibility
        split: str = "train",
        transform: Callable | None = None,
        size: int = 256,
        num_points: int = 2048,
        max_objects: int = 3,
        area: float = 10.0,
    ) -> None:
        self.split = split
        self.transform = transform
        self.size = size
        self.num_points = num_points
        self.max_objects = max_objects
        self.area = area
        self._offset = _SPLIT_OFFSETS.get(split, 3) * 1_000_003

    @property
    def num_classes(self) -> int:
        return len(self.CLASSES)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        gen = torch.Generator().manual_seed(self._offset + index)
        half = self.area / 2

        n_ground = self.num_points // 2
        ground = torch.rand(n_ground, 3, generator=gen)
        ground[:, :2] = (ground[:, :2] * 2 - 1) * half
        ground[:, 2] = torch.randn(n_ground, generator=gen) * 0.02

        num_obj = int(torch.randint(1, self.max_objects + 1, (1,), generator=gen))
        per_obj = max((self.num_points - n_ground) // num_obj, 1)

        boxes, labels, clusters = [], [], []
        for _ in range(num_obj):
            cx = float(torch.empty(1).uniform_(-half + 1, half - 1, generator=gen))
            cy = float(torch.empty(1).uniform_(-half + 1, half - 1, generator=gen))
            dx = float(torch.empty(1).uniform_(0.8, 2.0, generator=gen))
            dy = float(torch.empty(1).uniform_(0.8, 2.0, generator=gen))
            dz = float(torch.empty(1).uniform_(0.8, 2.0, generator=gen))
            cz = dz / 2  # resting on the ground plane

            pts = torch.rand(per_obj, 3, generator=gen) - 0.5
            pts[:, 0] = pts[:, 0] * dx + cx
            pts[:, 1] = pts[:, 1] * dy + cy
            pts[:, 2] = pts[:, 2] * dz + cz
            clusters.append(pts)
            boxes.append([cx, cy, cz, dx, dy, dz, 0.0])
            labels.append(0)

        points = torch.cat([ground, *clusters], dim=0)
        choice = torch.randint(0, len(points), (self.num_points,), generator=gen)
        points = points[choice]

        target = {
            "boxes_3d": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        if self.transform is not None:
            points = self.transform(points)
        return points, target


# ---------------------------------------------------------------------------
# File loaders + folder dataset
# ---------------------------------------------------------------------------


def load_point_cloud(path: str | Path) -> torch.Tensor:
    """Load an (N, 3+F) point cloud from ``.npy/.npz/.ply/.off/.xyz/.pts/.txt``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        arr = np.load(path)["points"]
    elif suffix in (".xyz", ".pts", ".txt"):
        arr = np.loadtxt(path, dtype=np.float32)
    elif suffix == ".off":
        arr = _load_off(path)
    elif suffix == ".ply":
        arr = _load_ply(path)
    else:
        raise ValueError(f"Unsupported point-cloud format: {suffix!r}")
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def _load_off(path: Path) -> np.ndarray:
    with open(path) as f:
        header = f.readline().strip()
        if header != "OFF":
            # Some OFF files put counts on the header line: "OFF 100 200 0"
            counts = header[3:].split()
        else:
            counts = f.readline().strip().split()
        num_verts = int(counts[0])
        verts = [list(map(float, f.readline().split()[:3])) for _ in range(num_verts)]
    return np.asarray(verts, dtype=np.float32)


def _load_ply(path: Path) -> np.ndarray:
    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "plyfile is required to read .ply point clouds. "
            "Install it with: pip install 'image-analytics[3d]'"
        ) from exc
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)


@DATASETS.register("pointcloud_folder")
class PointCloudFolder(Dataset):
    """Class-per-subdirectory point-cloud files::

        root/{split}/{class_name}/*.npy|.ply|.off|...

    Each cloud is resampled (random choice with replacement) to ``num_points``.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Callable | None = None,
        num_points: int = 1024,
        extensions: tuple[str, ...] = (".npy", ".npz", ".ply", ".off", ".xyz", ".pts", ".txt"),
    ) -> None:
        self.transform = transform
        self.num_points = num_points

        split_dir = Path(root) / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        if not class_dirs:
            raise FileNotFoundError(f"No class subdirectories under {split_dir}")
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        exts = {e.lower() for e in extensions}
        self.samples: list[tuple[Path, int]] = []
        for class_dir in class_dirs:
            label = self.class_to_idx[class_dir.name]
            for p in sorted(class_dir.rglob("*")):
                if p.suffix.lower() in exts:
                    self.samples.append((p, label))
        if not self.samples:
            raise FileNotFoundError(f"No point-cloud files under {split_dir}")

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def targets(self) -> list[int]:
        return [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        points = load_point_cloud(path)[:, :3]
        choice = torch.randint(0, len(points), (self.num_points,))
        points = points[choice]
        if self.transform is not None:
            points = self.transform(points)
        return points, label
