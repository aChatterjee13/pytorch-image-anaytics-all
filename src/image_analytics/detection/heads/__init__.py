"""Detection heads. Importing populates MODELS with detectors."""

from image_analytics.detection.heads.detr import DETR  # noqa: F401  (registration)
from image_analytics.detection.heads.faster_rcnn import FasterRCNN  # noqa: F401  (registration)
from image_analytics.detection.heads.fcos import FCOS  # noqa: F401  (registration)
from image_analytics.detection.heads.retinanet import RetinaNet  # noqa: F401  (registration)
