"""Multimodal Atlas implementation."""

import torch
import numpy as np
from transformers import AutoModel, AutoConfig
from torchvision.transforms import Compose, Normalize
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import Any, Dict, Optional
from easydict import EasyDict
from pathlib import Path
import yaml
import json
from huggingface_hub import hf_hub_download
from pillar.utils.logging import logger


def setup_device(device_spec: Optional[str] = None) -> torch.device:
    """
    Set up and validate device for model/data operations.

    Args:
        device_spec: Device specification ('cuda', 'cpu', 'cuda:0', etc.)

    Returns:
        Torch device object
    """
    if device_spec is None:
        device_spec = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_spec)

    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")

    return device


def load_model_config(model_name: str) -> Dict[str, Any]:
    """
    Load model-specific configuration.

    Args:
        model_name: Name of the model (e.g., 'medgemma', 'medimageinsights')

    Returns:
        Model configuration dictionary
    """
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "models" / f"{model_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    args = EasyDict(config)
    return args


def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get value from config using dot notation (e.g., 'data.root_dir').

    Args:
        config: Configuration dictionary
        key: Dot-separated key path
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    keys = key.split(".")
    value = config

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value


READOUT_LAYOUTS = ("legacy", "spatial")


def tokens_to_grid(feats_for_head, multiscale_layout, res, readout_layout="legacy"):
    """
    Arrange each scale's tokens on a 3D grid, resize every scale to `res` and
    concatenate them along channels (the "activ" output).

    Atlas windows the patch-embedded volume (B, C, A0, A1, A2) as
    "b c (g0 m0) (g1 m1) (g2 m2) -> (b g0 g1 g2) (m0 m1 m2) c", so after
    "(b nw) k c -> b (nw k) c" the token index is ordered (g0 g1 g2 m0 m1 m2),
    with (g0, g1, g2) = layout["grid_size"] and (m0, m1, m2) = layout["window_dims"].

    "spatial" inverts that exactly: activ[..., i, j, k] is the token at
    position (i, j, k) of the input axes (A0, A1, A2), and every fine-scale
    token is kept.

    "legacy" is the readout Pillar0-Sybil-1.5 was trained with. It unpacks the
    index as (g2 g0 g1 m2 m0 m1), which is not the order the tokens were written
    in, and resizes the resulting grid with nearest interpolation. For a
    256^3 chest CT, half of the fine-scale tokens never reach activ (only even
    planes of A0 survive) and each output plane mixes halves of different
    input planes. Keep it to reproduce the released checkpoints.

    Args:
        feats_for_head: list of (B, N, C) tensors, one per scale
        multiscale_layout: output of prepare_multiscale_layout
        res: finest token grid, [image_size[i] // patch_size[i] for i in range(3)]
        readout_layout: "legacy" or "spatial"

    Returns:
        (B, sum(C), *res) tensor
    """
    if readout_layout not in READOUT_LAYOUTS:
        raise ValueError(f"readout_layout must be one of {READOUT_LAYOUTS}, got {readout_layout!r}")
    if readout_layout == "spatial":
        pattern = "b (g0 g1 g2 m0 m1 m2) c -> b c (g0 m0) (g1 m1) (g2 m2)"
    else:
        pattern = "b (g2 g0 g1 m2 m0 m1) c -> b c (g2 m2) (g0 m0) (g1 m1)"

    outs = []
    for layout, scale in zip(multiscale_layout, feats_for_head):
        m0, m1, m2 = layout["window_dims"]
        g0, g1, g2 = layout["grid_size"]
        scale = rearrange(scale, pattern, g0=g0, g1=g1, g2=g2, m0=m0, m1=m1, m2=m2)
        # interpolate to the same size
        scale = nn.functional.interpolate(scale, size=res, mode="nearest")
        outs.append(scale)

    return torch.cat(outs, dim=1)


class MultimodalAtlas(nn.Module):
    """Multimodal Atlas medical image analysis."""

    def __init__(
        self,
        args,
        device="cuda",
        model_repo_id=None,
        model_revision=None,
        pretrained=False,
        readout_layout="legacy",
    ):
        super().__init__()
        if readout_layout not in READOUT_LAYOUTS:
            raise ValueError(f"readout_layout must be one of {READOUT_LAYOUTS}, got {readout_layout!r}")
        self.args = args
        self.readout_layout = readout_layout

        # Load model-specific config and setup device
        self.device = setup_device(device)
        self.model_repo_id = model_repo_id
        self.model_revision = model_revision
        self.pretrained = pretrained

        # Setup model
        self.setup_model()

        logger.info(f"Initialized MultimodalAtlas on device: {self.device}")

    def setup_model(self) -> None:
        """Initialize the model architecture and load pretrained weights."""

        print(f"Loading {self.model_repo_id} with revision {self.model_revision} from HuggingFace")
        logger.info("Loading model from HuggingFace")

        # self.model is CLIPMultimodalAtlas
        self.model = AutoModel.from_pretrained(self.model_repo_id, revision=self.model_revision, trust_remote_code=True)
        # self.visual is MultiModalAtlas
        if hasattr(self.model.model, "visual"):
            self.visual = self.model.model.visual
        else:
            self.visual = self.model.model

        self.model.to(self.device)

        # Get hidden_dim from the model - check various possible attributes
        if hasattr(self.visual, "embed_dim"):
            self.hidden_dim = self.visual.embed_dim
        else:
            # Default fallback
            self.hidden_dim = 1152
            logger.warning(f"Could not determine hidden_dim from model, using default: {self.hidden_dim}")

        self.model.train()

        logger.info(f"Model loaded successfully on device: {self.device}")

    def preprocess_single(self, image):
        """
        Preprocess a single exam for your model (for use in dataset __getitem__).

        Argscn:
            image: Tensor from dataset (normalized to [0,1] range)

        Returns:
            Preprocessed tensor ready for your model
        """
        ## hard
        # normalize_mean = self.normalize_mean
        # normalize_std = self.normalize_std
        # self.transform = Compose(
        #     [
        #         Normalize(
        #             mean=normalize_mean, std=normalize_std
        #         ),  # Normalize for single channel
        #     ]
        # )

        # image = self.transform(image)
        return image

    def extract_features(self, inputs: torch.Tensor, modality="chest_ct") -> np.ndarray:
        """
        Extract features from input images/volumes.

        Args:
            inputs: Input tensor of shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D

        Returns:
            Feature embeddings as numpy array of shape (B, feature_dim)
        """
        inputs = inputs.to(self.device)
        inputs_as_dict = {modality: inputs}

        with torch.no_grad():
            # Check if we have the HuggingFace model with extract_vision_feats method
            if hasattr(self.model, "extract_vision_feats"):
                features = self.model.extract_vision_feats(inputs)
            else:
                # Fallback to forward pass for feature extraction
                output = self.forward(inputs)
                if isinstance(output, dict) and "features" in output:
                    features = output["features"]
                elif isinstance(output, torch.Tensor):
                    features = output
                else:
                    raise ValueError(f"Cannot extract features from model output of type {type(output)}")

            # Convert to numpy for multiprocessing compatibility
            features = features.cpu().numpy().astype(np.float32)

        return features

    def eval(self):
        """Set model to evaluation mode."""
        if hasattr(self, "model"):
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor, batch=None) -> dict:
        visual = self.visual

        modality, image = batch["anatomy"][0], x
        bsz = image.shape[0]
        logger.debug(f"Entering MultiModalAtlas forward with modality: {modality}, batch size: {bsz}")
        x, grid_sizes = visual.build_scales({modality: x}, batch)

        modal_config = visual.model_config["modalities"][modality]
        image_size = modal_config["image_size"]
        merge_ratio = modal_config["merge_ratio"]
        local2global = modal_config["local2global"]
        patch_size = modal_config["patch_size"]
        multiscale_layout = visual.prepare_multiscale_layout(
            img_size=image_size,
            merge_ratio=merge_ratio,
            local2global=local2global,
            patch_size=patch_size,
        )
        grid_sizes = [layout["grid_size"] for layout in multiscale_layout]

        multiscale_feats = []
        for level, atlas_model in enumerate(visual.atlas_models):
            is_last = len(x) == 1 or level == len(visual.atlas_models) - 1
            x = atlas_model(
                x,
                grid_sizes=grid_sizes[level:],
                multiscale_layout=multiscale_layout[level:],
                merge_ratio=merge_ratio,
                local2global=local2global,
                modality=modality,
            )
            if not is_last:
                multiscale_feats.append(x[0])
                x = x[1:]

        multiscale_feats.append(x[0])

        feats_for_head = []
        for i, scale_tokens in enumerate(multiscale_feats):
            # Need to know if scale_tokens is windowed (B*NW, K, C) or flattened (B, N, C)
            # Assuming it's (B*NW, K, C) if NW > 1 based on layout
            if math.prod(grid_sizes[i]) > 1 and len(scale_tokens.shape) == 3:  # Check if likely windowed format
                rearranged_scale = rearrange(scale_tokens, "(b nw) k c -> b (nw k) c", b=bsz)
                logger.debug(f"  Rearranging scale {i} (windowed) {scale_tokens.shape} -> {rearranged_scale.shape}")
                feats_for_head.append(rearranged_scale)
            elif len(scale_tokens.shape) == 3 and scale_tokens.shape[0] == bsz:  # Already (B, N, C)
                logger.debug(f"  Using scale {i} (already BNC) shape: {scale_tokens.shape}")
                feats_for_head.append(scale_tokens)
            else:
                logger.warning(f"  Unexpected shape for scale {i}: {scale_tokens.shape}. Attempting BNC rearrange.")
                # Try a generic rearrange, might fail if batch dim isn't divisible
                try:
                    rearranged_scale = rearrange(scale_tokens, "(b nw) k c -> b (nw k) c", b=bsz)
                    feats_for_head.append(rearranged_scale)
                except Exception as e:
                    logger.error(f"    Failed to rearrange scale {i}: {e}. Skipping this scale for readout.")
                    continue

        if visual.multiscale_feats:
            x = []
            for i, scale in enumerate(feats_for_head):
                feats = visual.maxpool(scale.transpose(1, 2)).squeeze(2)
                x.append(feats)
            x = torch.cat(x, dim=1)
        else:
            ## return maxpool on last scale features only
            x = feats_for_head[0]
            x = visual.maxpool(x.transpose(1, 2)).squeeze(2)

        import torch.nn.functional as F

        res = [image_size[i] // patch_size[i] for i in range(3)]
        outs = tokens_to_grid(feats_for_head, multiscale_layout, res, self.readout_layout)

        outputs = {
            "activ": outs,
            # could also consider taking scale 2 and scale 3 and interpolate up then concat along channels
            # can modify atlas for the cleanest API
            "pooled": F.normalize(x, dim=-1),
        }
        return outputs
