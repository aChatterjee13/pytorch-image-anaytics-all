"""3D perception: pure-PyTorch point ops, PointNet / PointNet++ / DGCNN
(classification + part segmentation), PointPillars (3D detection), and 3D box
utilities. CUDA-only methods (SECOND, CenterPoint, BEVFormer, Mask3D) are
lazy-gated wrappers in ``voxel`` / ``bev`` / ``segmentation_3d``.

Importing the model modules populates MODELS.
"""

from image_analytics.detection_3d import (  # noqa: F401  (registration side effects)
    bev,
    dgcnn,
    pointnet,
    pointpillars,
    segmentation_3d,
    voxel,
)
