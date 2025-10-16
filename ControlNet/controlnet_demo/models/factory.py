import logging, torch
from typing import Any, Dict, Tuple

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

class ControlNetBundle:
    def __init__(self, model, sampler):
        self.model = model
        self.sampler = sampler


def create_controlnet_model(cfg: Dict[str, Any]) -> ControlNetBundle:
    """Create ControlNet (SD1.5 backbone) + DDIM sampler.
    Expect keys: config_path, weights_path, device
    """
    conf = cfg["controlnet"]
    model = create_model(conf["config_path"]).cpu()
    # location='cuda' keeps compatibility with existing checkpoints
    sd = load_state_dict(conf["weights_path"], location=conf.get("device", "cuda"))
    model.load_state_dict(sd, strict=False)
    device = conf.get("device", "cuda")
    model = model.to(device)
    sampler = DDIMSampler(model)
    logging.info("Model loaded: %s | Weights: %s", conf["config_path"], conf["weights_path"])
    return ControlNetBundle(model, sampler)