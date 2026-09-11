# src/encoders.py
import torch
import torch.nn as nn
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

# Band order for the v0.2.0 checkpoints, read from
# configilm.extra.BENv2_utils.STANDARD_BANDS. NOT interchangeable with v0.1.1.
S1_BANDS = ["VV", "VH"]
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

REPOS = {
    "s1": "BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.2.0",
    "s2": "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
}
BANDS = {"s1": S1_BANDS, "s2": S2_BANDS}


class FrozenEncoder(nn.Module):
    """Pretrained BigEarthNet ResNet50 trunk, frozen, returning a spatial feature grid."""

    def __init__(self, modality: str, image_size: int = 120):
        super().__init__()
        assert modality in REPOS, f"modality must be one of {list(REPOS)}"
        self.modality = modality
        self.expected_bands = BANDS[modality]
        self.image_size = image_size

        classifier = BigEarthNetv2_0_ImageClassifier.from_pretrained(REPOS[modality])
        self.backbone = classifier.model.vision_encoder
        self.feature_dim = self.backbone.num_features  # 2048 for resnet50

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        """Keep the frozen trunk in eval mode even when the parent model trains."""
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"expected (B, C, H, W), got {tuple(x.shape)}"
        assert x.shape[1] == len(self.expected_bands), (
            f"{self.modality}: expected {len(self.expected_bands)} bands "
            f"in order {self.expected_bands}, got {x.shape[1]}"
        )
        assert x.shape[-2:] == (self.image_size, self.image_size), (
            f"{self.modality}: expected {self.image_size}x{self.image_size}, "
            f"got {tuple(x.shape[-2:])}"
        )
        return self.backbone.forward_features(x)  # (B, 2048, 4, 4)