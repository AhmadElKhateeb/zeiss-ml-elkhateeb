import os, argparse, yaml, logging

import imageio
import numpy as np
import matplotlib.pyplot as plt

from ldm.modules.image_degradation.utils_image import imshow
from utils.logging_utils import setup_logging
from utils.repro import set_seed, env_snapshot, create_run_dir, dump_json
from data.loader import ImageSpec, ImageLoader
from models.factory import create_controlnet_model
from inference.pipeline import InferencePipeline, SamplerCfg


def parse_args():
    ap = argparse.ArgumentParser(description="ControlNet demo refactored")
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    ap.add_argument("--input", type=str, default="test_imgs/mri_brain.jpg")
    ap.add_argument("--save", default=True, help="Save outputs under run dir")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    run_dir = create_run_dir(cfg.get("output_root", "runs"), cfg.get("run_name", "run"))
    setup_logging(run_dir)

    # Snapshot config & env
    dump_json(cfg, os.path.join(run_dir, "config_snapshot.json"))
    dump_json(env_snapshot(), os.path.join(run_dir, "env_snapshot.json"))

    set_seed(cfg.get("seed", 0))

    # Load data
    loader = ImageLoader(ImageSpec(args.input, cfg["image_resolution"]))
    original_np, resized_np = loader.load()

    # Model
    bundle = create_controlnet_model(cfg["model"])

    # Pipeline
    s = cfg["sampler"]
    sampler_cfg = SamplerCfg(ddim_steps=s["ddim_steps"], scale=s["scale"], eta=s["eta"],
                             guess_mode=s["guess_mode"], strength=s["strength"])

    cond = cfg["conditioning"]
    pipe = InferencePipeline(bundle, sampler_cfg, cond["type"], cond.get("canny", {}))

    prompts = {
        "positive": cfg["prompt"]["positive"],
        "positive_suffix": cfg["prompt"].get("positive_suffix", ""),
        "negative": cfg["prompt"]["negative"],
    }

    logging.info("Starting inference …")
    results = pipe.run(original_np, resized_np, prompts, num_samples=cfg["num_samples"])

    # Simple visualization
    n = len(results) + 1 # results and canny input (+1)
    fig, axs = plt.subplots(1, n, figsize=(5*n, 5))

    axs[0].imshow(imageio.imread(args.input))
    for i, img in enumerate(results, start=1):
        axs[i].imshow(img)
        axs[i].axis("off")
    plt.tight_layout()

    if args.save:
        out_dir = os.path.join(run_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"result.png")
        plt.savefig(path)
        # for i, img in enumerate(results):
        #     path = os.path.join(out_dir, f"result.png")
        #     imageio.imwrite(path, fig)
        logging.info("Saved %d outputs -> %s", len(results), out_dir)
    else:
        plt.show()


if __name__ == "__main__":
    main()
