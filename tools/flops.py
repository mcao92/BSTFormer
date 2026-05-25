import argparse
import os.path as osp
import sys
import time

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import torch
from thop import profile

from bstformer.models import build_model
from bstformer.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device = "cpu"
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    device = torch.device(args.device)
    profile_cfg = cfg.get("profile", {})

    height = profile_cfg.get("height", 256)
    width = profile_cfg.get("width", 256)
    frames = profile_cfg.get("frames", 8)
    repeat = profile_cfg.get("repeat", 10)

    model = build_model(cfg.model).to(device).eval()
    meas = torch.randn(1, 1, height, width).to(device)
    Phi = torch.randn(1, frames, height, width).to(device)
    Phi_s = torch.randn(1, 1, height, width).to(device)

    macs, params = profile(model, inputs=(meas, Phi, Phi_s))
    with torch.no_grad():
        for _ in range(repeat):
            start = time.time()
            model(meas, Phi, Phi_s)
            if device.type == "cuda":
                torch.cuda.synchronize()
            print("time: ", time.time() - start)
    print("para: {} M, FLOPs: {} G.".format(params / 1e6, macs / 1e9))


if __name__ == "__main__":
    main()
