"""Face ReID package — frozen DINOv2 + regional pooling."""

from cvi.face_id.config import FaceIDConfig, FaceIDTrainConfig
from cvi.face_id.dataset import FaceReIDDataset, build_dogface_dataset
from cvi.face_id.model import FaceIDModel, FaceRegionalEncoder
from cvi.face_id.losses import FaceIDObjective
from cvi.face_id.sampler import FaceReIDSampler, PositiveStrength
from cvi.face_id.trainer import (
    build_faceid_model,
    build_faceid_optimizer,
    train_faceid_epoch,
)
from cvi.face_id.evaluation import extract_face_embeddings, evaluate_face_retrieval
from cvi.face_id.types import AlignedFace, FaceIDOutput

__all__ = [
    "AlignedFace",
    "FaceIDConfig",
    "FaceIDModel",
    "FaceIDObjective",
    "FaceIDOutput",
    "FaceIDTrainConfig",
    "FaceReIDDataset",
    "FaceReIDSampler",
    "FaceRegionalEncoder",
    "PositiveStrength",
    "build_dogface_dataset",
    "build_faceid_model",
    "build_faceid_optimizer",
    "evaluate_face_retrieval",
    "extract_face_embeddings",
    "train_faceid_epoch",
]
