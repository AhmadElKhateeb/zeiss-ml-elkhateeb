import os, pytest, numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_YAML = os.path.join(REPO_ROOT, "models", "cldm_v15.yaml")
MODEL_WEIGHTS = os.path.join(REPO_ROOT, "models", "control_sd15_canny.pth")

skip_missing = pytest.mark.skipif(
    not (os.path.isfile(MODEL_YAML) and os.path.isfile(MODEL_WEIGHTS)),
    reason="ControlNet weights/config not available"
)

@skip_missing
def test_smoke(monkeypatch):
    from models.factory import create_controlnet_model
    from inference.pipeline import InferencePipeline, SamplerCfg

    bundle = create_controlnet_model({
        "controlnet": {
            "config_path": MODEL_YAML,
            "weights_path": MODEL_WEIGHTS,
            "device": "cpu"
        }
    })

    sampler_cfg = SamplerCfg(ddim_steps=1, scale=1.0, eta=0.0, guess_mode=False, strength=1.0)
    pipe = InferencePipeline(bundle, sampler_cfg, "canny", {"low_threshold": 50, "high_threshold": 100})

    dummy = np.zeros((512, 512, 3), dtype=np.uint8)
    out = pipe.run(dummy, dummy, {"positive":"mri", "positive_suffix":"test", "negative":""}, num_samples=1)
    assert isinstance(out, list) and len(out) == 2
