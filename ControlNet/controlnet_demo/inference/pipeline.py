from dataclasses import dataclass
from typing import Any, Dict, List
import logging
import numpy as np
import torch, einops

from annotator.util import HWC3
from annotator.canny import CannyDetector

from utils.registry import CONDITIONERS


# ---- Conditioners ---------------------------------------------------------
@CONDITIONERS.register("canny")
class CannyConditioner:
    def __init__(self, low_threshold: int, high_threshold: int):
        self.detector = CannyDetector()
        self.low = low_threshold
        self.high = high_threshold

    def __call__(self, image_np: np.ndarray) -> np.ndarray:
        edges = self.detector(HWC3(image_np), self.low, self.high)
        return HWC3(edges)


@dataclass
class SamplerCfg:
    ddim_steps: int
    scale: float
    eta: float
    guess_mode: bool
    strength: float


class InferencePipeline:
    def __init__(self, model_bundle, sampler_cfg: SamplerCfg, conditioner_name: str, conditioner_kwargs: Dict[str, Any]):
        self.model = model_bundle.model
        self.sampler = model_bundle.sampler
        Conditioner = CONDITIONERS.get(conditioner_name)
        self.conditioner = Conditioner(**conditioner_kwargs)
        self.sampler_cfg = sampler_cfg

    def _prepare_condition(self, cond_np: np.ndarray, num_samples: int) -> torch.Tensor:
        control = torch.from_numpy(cond_np.copy()).float().to(self.model.device) / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        return control

    def run(self, input_np: np.ndarray, resized_np: np.ndarray, prompts: Dict[str, str], num_samples: int) -> List[np.ndarray]:
        with torch.no_grad():
            H, W, _ = resized_np.shape
            cond_map = self.conditioner(resized_np)
            control = self._prepare_condition(cond_map, num_samples)

            pos = f"{prompts['positive']}, {prompts.get('positive_suffix','').strip()}".strip(', ')
            neg = prompts['negative']

            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([pos] * num_samples)]}
            un_cond = {"c_concat": None if self.sampler_cfg.guess_mode else [control],
                       "c_crossattn": [self.model.get_learned_conditioning([neg] * num_samples)]}

            shape = (4, H // 8, W // 8)

            # Control strength schedule (matches original demo semantics)
            if self.sampler_cfg.guess_mode:
                scales = [self.sampler_cfg.strength * (0.825 ** float(12 - i)) for i in range(13)]
            else:
                scales = [self.sampler_cfg.strength] * 13
            self.model.control_scales = scales

            samples, _ = self.sampler.sample(
                self.sampler_cfg.ddim_steps, num_samples, shape, cond, verbose=False,
                eta=self.sampler_cfg.eta, unconditional_guidance_scale=self.sampler_cfg.scale,
                unconditional_conditioning=un_cond,
            )

            x_samples = self.model.decode_first_stage(samples)
            x = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5)
            x = x.clamp(0, 255).to(torch.uint8).cpu().numpy()
            results = [x[i] for i in range(num_samples)]
            # Return conditioning visualization first for parity with old demo
            return [255 - cond_map] + results
