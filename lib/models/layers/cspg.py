from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from lib.models.layers.clip import clip


COCO_CATEGORIES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)


class SharedAdapter(nn.Module):
    """Lightweight residual adapter shared by CLIP image and text features."""

    def __init__(self, dim: int = 512, reduction: int = 4) -> None:
        super().__init__()
        hidden_dim = dim // reduction
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc(x)


class CategorySemanticPromptGenerator(nn.Module):
    """Generate a template-aware category-level semantic prompt with CLIP ViT-B/32."""

    def __init__(self, clip_name: str = "ViT-B/32", clip_dim: int = 512, token_dim: int = 768) -> None:
        super().__init__()
        # Load CLIP on CPU first. The enclosing tracker later moves the complete
        # module to the correct training or inference device.
        device = "cpu"
        self.clip_model, self.preprocess = clip.load(clip_name, device=device)
        self.clip_model.eval()
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

        self.shared_adapter = SharedAdapter(clip_dim)
        self.convert_vector = nn.Linear(clip_dim, token_dim)
        self.categories = COCO_CATEGORIES

        text_inputs = torch.cat([clip.tokenize(f"a photo of {category}") for category in self.categories]).to(device)
        with torch.no_grad():
            raw_text_features = self.clip_model.encode_text(text_inputs).float()
            raw_text_features = raw_text_features / raw_text_features.norm(dim=-1, keepdim=True)
        self.register_buffer("raw_text_features", raw_text_features, persistent=False)

        self._external_template: Optional[Any] = None
        self._cached_prompt: Optional[torch.Tensor] = None
        self._last_info: Dict[str, Any] = {
            "selected_class": None,
            "selected_class_index": -1,
            "semantic_confidence": -1.0,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        # CLIP remains frozen and in evaluation mode even when the tracker is trained.
        self.clip_model.eval()
        return self

    def reset_template_cache(self) -> None:
        self._cached_prompt = None
        self._last_info = {
            "selected_class": None,
            "selected_class_index": -1,
            "semantic_confidence": -1.0,
        }

    def set_external_template(self, template_patch: Any) -> None:
        """Set the raw RGB template used during online inference and clear the old prompt cache."""
        self._external_template = template_patch
        self.reset_template_cache()

    def get_last_cspg_info(self) -> Dict[str, Any]:
        return dict(self._last_info)

    @staticmethod
    def _to_pil(image: Any):
        if isinstance(image, np.ndarray):
            array = image
            if array.ndim == 3 and array.shape[-1] >= 3:
                array = array[..., :3]
            return transforms.ToPILImage()(array)
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 4:
                tensor = tensor[0]
            if tensor.shape[0] > 3:
                tensor = tensor[:3]
            return transforms.ToPILImage()(tensor)
        return image

    def _encode_single_prompt(
        self,
        template_image: Any,
        adapted_text_features: torch.Tensor,
    ) -> torch.Tensor:
        device = adapted_text_features.device
        template_pil = self._to_pil(template_image)
        image_input = self.preprocess(template_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            raw_image_features = self.clip_model.encode_image(image_input).float()
            raw_image_features = raw_image_features / raw_image_features.norm(dim=-1, keepdim=True)

        adapted_image_features = self.shared_adapter(raw_image_features)
        probabilities = (100.0 * adapted_image_features @ adapted_text_features.T).softmax(dim=-1)
        confidence, index = probabilities[0].max(dim=0)
        selected_text_feature = adapted_text_features[index].unsqueeze(0)

        self._last_info = {
            "selected_class": self.categories[int(index.item())],
            "selected_class_index": int(index.item()),
            "semantic_confidence": float(confidence.detach().cpu().item()),
        }

        # Preserve the behavior used by the original training implementation.
        return self.convert_vector(selected_text_feature).softmax(dim=-1)

    def forward(self, template: torch.Tensor, search: Optional[torch.Tensor] = None) -> torch.Tensor:
        del search  # Search images are not required for category-prompt selection.
        adapted_text_features = self.shared_adapter(self.raw_text_features.float())

        if not self.training and self._cached_prompt is not None:
            batch_size = int(template.shape[0]) if isinstance(template, torch.Tensor) and template.ndim == 4 else 1
            return self._cached_prompt.expand(batch_size, -1, -1)

        if not self.training and self._external_template is not None:
            prompt = self._encode_single_prompt(self._external_template, adapted_text_features).reshape(1, 1, -1)
            self._cached_prompt = prompt.detach()
            return prompt

        if not isinstance(template, torch.Tensor):
            prompt = self._encode_single_prompt(template, adapted_text_features)
            return prompt.reshape(1, 1, -1)

        batch_size = template.shape[0] if template.ndim == 4 else 1
        images = template if template.ndim == 4 else template.unsqueeze(0)
        prompts = [self._encode_single_prompt(images[i], adapted_text_features) for i in range(batch_size)]
        return torch.cat(prompts, dim=0).reshape(batch_size, 1, -1)


# Backward-compatible class alias for old checkpoints and external imports.
Category_embedding = CategorySemanticPromptGenerator
