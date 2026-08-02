"""Face ReID package — frozen DINOv2 + regional pooling."""

from identity_methods.face.config import FaceIDConfig, FaceIDTrainConfig
from identity_methods.face.dataset import (
    FaceReIDDataset,
    RoiFaceReIDDataset,
    build_dogface_dataset,
)
from experiments.face_evaluation import evaluate_face_retrieval, extract_face_embeddings
from identity_methods.face.losses import FaceIDObjective
from identity_methods.face.model import FaceIDModel, FaceRegionalEncoder
from identity_methods.face.sampler import FaceReIDSampler, PositiveStrength
from identity_methods.face.trainer import (
    build_faceid_model,
    build_faceid_optimizer,
    train_faceid_epoch,
)
from identity_methods.face.types import AlignedFace, FaceIDOutput

__all__ = [
    "AlignedFace",
    "FaceIDConfig",
    "FaceIDModel",
    "FaceIDObjective",
    "FaceIDOutput",
    "FaceIDTrainConfig",
    "FaceReIDDataset",
    "RoiFaceReIDDataset",
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
