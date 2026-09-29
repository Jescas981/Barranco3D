import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from ..utils.base_model import BaseModel


MAST3R_ROOT = (
    Path(__file__).resolve().parents[2]
    / "third_party/aerial-megadepth/mast3r"
)

sys.path.insert(0, str(MAST3R_ROOT))

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import extract_correspondences_nonsym
# from hloc_viz import plot_matches, plot_images

# ------------------------------------------------------------
# Per-weights checkpoint source.
#
# "aerialmd" -> aerial-finetuned MASt3R checkpoint (used by the
#     "mast3ramd" conf in hloc.extract_match_features).
# "mast3r"   -> the standard/generic MASt3R checkpoint (used by
#     the "mast3r" conf in hloc.extract_match_features).
#
# Add new entries here if you introduce new "weights" options
# in hloc.extract_match_features.confs.
# ------------------------------------------------------------
WEIGHTS_SOURCES = {
    "aerialmd": {
        "repo_id": "kvuong2711/checkpoint-aerial-mast3r",
        "dirname": "checkpoint-aerial-mast3r",
    },
    "mast3r": {
        "repo_id": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        "dirname": "checkpoint-mast3r",
    },
}

class Mast3r(BaseModel):

    required_inputs = [
        "image0",
        "image1",
    ]

    def _init(self, conf):

        weights = conf["weights"]

        if weights not in WEIGHTS_SOURCES:
            raise ValueError(
                f"Unknown weights {weights!r}. "
                f"Available options: {list(WEIGHTS_SOURCES.keys())}"
            )

        source = WEIGHTS_SOURCES[weights]

        checkpoint_dir = (
            Path(__file__).resolve().parents[2]
            / "third_party"
            / "aerial-megadepth"
            / "checkpoints"
            / source["dirname"]
        )

        if not checkpoint_dir.exists():

            print(
                f"Downloading MASt3R ({weights}) checkpoint to "
                f"{checkpoint_dir}..."
            )

            checkpoint_dir.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            snapshot_download(
                repo_id=source["repo_id"],
                repo_type="model",
                local_dir=checkpoint_dir,
            )

            print(f"MASt3R ({weights}) checkpoint downloaded.")

        else:
            print(
                f"Using existing MASt3R ({weights}) checkpoint: "
                f"{checkpoint_dir}"
            )

        self.model = (
            AsymmetricMASt3R
            .from_pretrained(checkpoint_dir)
            .eval()
        )

        self.subsample = 4
        self.pixel_tol = 3
        self.match_conf = 0.3

    def _forward(self, data):

        image0 = data["image0"]
        image1 = data["image1"]

        H0, W0 = image0.shape[-2:]
        H1, W1 = image1.shape[-2:]

        view1 = {
            "img": image0,
            "true_shape": torch.tensor(
                [[H0, W0]],
                device=image0.device,
            ),
            "instance": ["1"],
        }

        view2 = {
            "img": image1,
            "true_shape": torch.tensor(
                [[H1, W1]],
                device=image1.device,
            ),
            "instance": ["2"],
        }

        with torch.no_grad():
            pred1, pred2 = self.model(
                view1,
                view2,
            )

        device = image0.device


        corres = extract_correspondences_nonsym(
            pred1["desc"][0],
            pred2["desc"][0],
            pred1["desc_conf"][0].detach().cpu().numpy(),
            pred2["desc_conf"][0].detach().cpu().numpy(),
            device=device,
            subsample=self.subsample,
            pixel_tol=self.pixel_tol,
        )

        matches_im0 = corres[0]
        matches_im1 = corres[1]
        scores = corres[2]

        # Confidence filtering
        mask = scores >= self.match_conf

        matches_im0 = matches_im0[mask]
        matches_im1 = matches_im1[mask]
        scores = scores[mask]

        # ignore small border around the edge
        H0, W0 = view1['true_shape'][0]
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)
    
        H1, W1 = view2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)
    
        valid_matches = valid_matches_im0 & valid_matches_im1

        matches_im0 = matches_im0[valid_matches]
        matches_im1 = matches_im1[valid_matches]
        scores = scores[valid_matches]

        return {
            "keypoints0": matches_im0.float(),
            "keypoints1": matches_im1.float(),
            "scores": scores.float(),
        }