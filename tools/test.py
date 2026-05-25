import argparse
import os
import os.path as osp
import sys
import time

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from bstformer.datasets import build_dataset
from bstformer.models import build_model
from bstformer.utils.checkpoint import load_checkpoint
from bstformer.utils.config import Config
from bstformer.utils.image import save_image
from bstformer.utils.logger import Logger
from bstformer.utils.mask import build_mask
from bstformer.utils.metrics import compare_psnr, compare_ssim
from bstformer.models.network import A, At


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device = "cpu"
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    work_dir = args.work_dir or cfg.get("work_dir", osp.join("work_dirs", osp.splitext(osp.basename(args.config))[0]))
    weights = args.weights or cfg.get("checkpoints")
    if weights is None:
        raise ValueError("No checkpoint specified. Set cfg.checkpoints or pass --weights.")

    log_dir = osp.join(work_dir, "test_log")
    test_dir = osp.join(work_dir, "test_results_vis")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    logger = Logger(log_dir)

    device = torch.device(args.device)
    mask, mask_s = build_mask(cfg.test_mask)
    mask = mask.float()
    mask_s = mask_s.float()
    test_data = build_dataset(cfg.test_data, {"mask": mask})
    data_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    model = build_model(cfg.model).to(device)
    logger.info("Load checkpoint: {}".format(weights))
    load_checkpoint(model, weights, strict=args.strict, map_location=device)
    model.eval()

    psnr_dict, ssim_dict = {}, {}
    psnr_list, ssim_list = [], []
    start_time = time.time()
    for data_iter, data in enumerate(data_loader):
        meas, gt = data
        meas = meas[0].float().to(device).unsqueeze(1)
        gt = gt[0].float().numpy()
        batch_size, frames, height, width = gt.shape

        Phi = mask.to(device).expand([batch_size, frames, height, width])
        Phi_s = mask_s.to(device).expand([batch_size, 1, height, width])

        with torch.no_grad():
            xe = At(meas, Phi)
            yb = A(xe, Phi)
            xe = xe + At(torch.div(meas - yb, Phi_s), Phi)
            outputs = model(meas, Phi, Phi_s)
        if not isinstance(outputs, list):
            outputs = [outputs]
        out = outputs[-1].detach().cpu().numpy()
        xe = xe.detach().cpu().numpy()

        psnr, ssim = 0.0, 0.0
        data_name = test_data.data_name_list[data_iter]
        name_root = data_name.split("_")[0] if "_" in data_name else osp.splitext(data_name)[0]
        for ii in range(batch_size):
            for jj in range(frames):
                out_frame = out[ii, jj]
                gt_frame = gt[ii, jj]
                psnr += compare_psnr(gt_frame * 255, out_frame * 255)
                ssim += compare_ssim(gt_frame * 255, out_frame * 255)

            image_name = osp.join(test_dir, f"{name_root}_{ii}.png")
            save_image(out[ii], gt[ii], image_name)

        psnr = psnr / (batch_size * frames)
        ssim = ssim / (batch_size * frames)
        psnr_dict[name_root] = np.round(psnr, 4)
        ssim_dict[name_root] = np.round(ssim, 4)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        logger.info("{}, psnr: {:.4f}, ssim: {:.4f}.".format(name_root, psnr, ssim))

    logger.info("Average runtime: {:.4f}s.".format(time.time() - start_time))
    logger.info("psnr_mean: {:.4f}.".format(np.mean(psnr_list)))
    logger.info("ssim_mean: {:.4f}.".format(np.mean(ssim_list)))
    logger.info("psnr: {}.".format(psnr_dict))
    logger.info("ssim: {}.".format(ssim_dict))


if __name__ == "__main__":
    main()
