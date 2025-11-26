from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
import open_clip
from PIL import Image


class SigLipITM:
    def __init__(self, device: torch.device | str | None = None,
                 model_name: str = "ViT-B-16-SigLIP-384",
                 pretrained: str = "webli",
                 backend: str = "open_clip") -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.backend = backend
        if backend == "timm":
            import timm
            self.model = timm.create_model(model_name, pretrained=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            from timm.data import resolve_data_config, create_transform
            cfg = resolve_data_config({}, model=self.model)
            self.preprocess = create_transform(**cfg)
            self.tokenizer = None
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)

    def _to_pil(self, image_rgb: np.ndarray) -> Image.Image:
        if isinstance(image_rgb, Image.Image):
            return image_rgb
        img = image_rgb
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return Image.fromarray(img)

    @torch.no_grad()
    def encode_image(self, image_rgb: Union[np.ndarray, List[np.ndarray], Image.Image]) -> np.ndarray:
        if isinstance(image_rgb, list):
            if len(image_rgb) == 0:
                return np.empty((0, self.model.visual.output_dim), dtype=np.float32)
            pil_images = [self._to_pil(img) for img in image_rgb]
            img_t = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        else:
            pil_img = self._to_pil(image_rgb)
            img_t = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        if self.backend == "timm":
            if hasattr(self.model, "encode_image"):
                feats = self.model.encode_image(img_t)
            elif hasattr(self.model, "forward_image"):
                feats = self.model.forward_image(img_t)
            else:
                feats = self.model(img_t)
        else:
            feats = self.model.encode_image(img_t)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.detach().cpu().numpy()
        if isinstance(image_rgb, list):
            return feats
        return feats.squeeze(0)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        if self.backend == "timm":
            if hasattr(self.model, "encode_texts"):
                t_feat = self.model.encode_texts(texts)
                if isinstance(t_feat, np.ndarray):
                    t_feat = torch.from_numpy(t_feat).to(self.device)
            elif hasattr(self.model, "encode_text"):
                lst = [self.model.encode_text(t) for t in texts]
                t_feat = torch.stack([torch.tensor(o, device=self.device) if not isinstance(o, torch.Tensor) else o for o in lst], dim=0)
            else:
                raise NotImplementedError("Model has no text encoding method")
        else:
            toks = self.tokenizer(texts).to(self.device)
            t_feat = self.model.encode_text(toks)
        
        t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        return t_feat.detach().cpu().numpy()

    @torch.no_grad()
    def image_text_scores(self, image_rgb: np.ndarray | Image.Image, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        img = self._to_pil(image_rgb)
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        if self.backend == "timm":
            if hasattr(self.model, "encode_image"):
                i_feat = self.model.encode_image(img_t)
            elif hasattr(self.model, "forward_image"):
                i_feat = self.model.forward_image(img_t)
            else:
                i_feat = self.model(img_t)
            if hasattr(self.model, "encode_texts"):
                t_feat = self.model.encode_texts(texts)
                if isinstance(t_feat, np.ndarray):
                    t_feat = torch.from_numpy(t_feat).to(self.device)
            elif hasattr(self.model, "encode_text"):
                lst = []
                for t in texts:
                    out = self.model.encode_text(t)
                    lst.append(out if isinstance(out, torch.Tensor) else torch.tensor(out, device=self.device))
                t_feat = torch.stack(lst, dim=0)
            else:
                raise NotImplementedError
        else:
            toks = self.tokenizer(texts).to(self.device)
            i_feat = self.model.encode_image(img_t)
            t_feat = self.model.encode_text(toks)
        i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
        t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        sims = (i_feat @ t_feat.T).squeeze(0).detach().cpu().numpy()
        return sims.astype(float)

    @torch.no_grad()
    def text_text_scores(self, texts_a: List[str], texts_b: List[str]) -> np.ndarray:
        if not texts_a or not texts_b:
            return np.array([])
        
        feats_a = self.encode_texts(texts_a)
        feats_b = self.encode_texts(texts_b)
        
        sims = (feats_a @ feats_b.T)
        return sims.squeeze().astype(float)

    def image_image_scores(self, image1: Image, image2: Union[Image, List[Image]]) -> np.ndarray:
        image1_processed = self.preprocess(image1).unsqueeze(0).to(self.device)

        # Handle both single image and list of images for the second argument
        if not isinstance(image2, list):
            image2 = [image2]
        
        # Preprocess all images in the list and stack them into a single batch tensor
        image2_processed = torch.stack([self.preprocess(img) for img in image2]).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image1_features = self.model.encode_image(image1_processed)
            image2_features = self.model.encode_image(image2_processed)
            image1_features /= image1_features.norm(dim=-1, keepdim=True)
            image2_features /= image2_features.norm(dim=-1, keepdim=True)
            scores = image1_features @ image2_features.T

        return scores.squeeze().cpu().numpy()


