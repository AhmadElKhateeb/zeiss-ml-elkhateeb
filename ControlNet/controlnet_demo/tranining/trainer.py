"""Minimal placeholder — keeps interfaces stable without adding deps.
Add dataset + optimization here later if training is needed.
"""
from dataclasses import dataclass

@dataclass
class TrainerConfig:
    lr: float = 1e-4

class Trainer:
    def __init__(self, model_bundle, cfg: TrainerConfig):
        self.model = model_bundle.model
        self.cfg = cfg

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Training loop not implemented in this demo.")
