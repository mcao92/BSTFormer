import argparse
import json
import os
import os.path as osp
import sys
import time

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import bstformer.datasets.davis_aug  # noqa: F401
from bstformer.datasets import build_dataset
from bstformer.models import build_model
from bstformer.utils.checkpoint import load_checkpoint
from bstformer.utils.config import Config
from bstformer.utils.image import save_image
from bstformer.utils.logger import Logger
from bstformer.utils.loss_builder import build_loss
from bstformer.utils.mask import build_mask
from bstformer.utils.optim_builder import build_optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", default=-1)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.distributed:
        args.device = torch.device("cuda", int(args.local_rank))
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    work_dir = args.work_dir or cfg.get("work_dir", osp.join("work_dirs", osp.splitext(osp.basename(args.config))[0]))
    if args.resume is not None:
        cfg.resume = args.resume

    log_dir = osp.join(work_dir, "log")
    show_dir = osp.join(work_dir, "show")
    train_image_dir = osp.join(work_dir, "train_images")
    checkpoints_dir = osp.join(work_dir, "checkpoints")
    for path in [log_dir, show_dir, train_image_dir, checkpoints_dir]:
        os.makedirs(path, exist_ok=True)

    logger = Logger(log_dir)
    writer = SummaryWriter(log_dir=show_dir)

    rank = 0
    if args.distributed:
        backend = "nccl" if dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()

    device = torch.device(args.device)
    model = build_model(cfg.model).to(device)
    optimizer = build_optimizer(cfg.optimizer, {"params": model.parameters()})
    criterion = build_loss(cfg.loss).to(device)

    start_epoch = 0
    if rank == 0:
        logger.info("cfg info:\n{}".format(json.dumps(cfg, indent=4)))
        if cfg.get("checkpoints") is not None:
            logger.info("Load checkpoint: {}".format(cfg.checkpoints))
            load_checkpoint(model, cfg.checkpoints, strict=False, map_location=device)
        if cfg.get("resume") is not None:
            logger.info("Resume from: {}".format(cfg.resume))
            resume_dict = torch.load(cfg.resume, map_location=device)
            start_epoch = resume_dict["epoch"] + 1
            model.load_state_dict(resume_dict["model_state_dict"])
            optimizer.load_state_dict(resume_dict["optim_state_dict"])

    if args.distributed:
        local_rank = int(args.local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    mask, mask_s = build_mask(cfg.train_mask)
    mask = mask.float()
    mask_s = mask_s.float()
    train_data = build_dataset(cfg.train_data, {"mask": mask.cpu()})
    sampler = DistributedSampler(train_data, shuffle=True) if args.distributed else None
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.data.samples_per_gpu,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.data.workers_per_gpu,
    )

    for epoch in range(start_epoch, cfg.runner.max_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        iter_num = len(train_loader)

        for iteration, data in enumerate(train_loader):
            gt, meas = data
            gt = gt.float().to(device)
            meas = meas.float().to(device).unsqueeze(1)
            batch_size, _, height, width = meas.shape
            frames = gt.shape[-3] if gt.ndim == 4 else gt.shape[-3]

            Phi = mask.to(device).expand([batch_size, frames, height, width])
            Phi_s = mask_s.to(device).expand([batch_size, 1, height, width])

            optimizer.zero_grad()
            outputs = model(meas, Phi, Phi_s)
            if not isinstance(outputs, list):
                outputs = [outputs]
            loss = torch.sqrt(criterion(outputs[-1], gt))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if rank == 0 and iteration % cfg.log_config.interval == 0:
                lr = optimizer.state_dict()["param_groups"][0]["lr"]
                logger.info(
                    "epoch: [{}][{}/{}], lr: {:.6f}, loss: {:.5f}.".format(
                        epoch, iteration, iter_num, lr, loss.item()
                    )
                )
                writer.add_scalar("loss", loss.item(), epoch * iter_num + iteration)

            if rank == 0 and iteration % cfg.save_image_config.interval == 0:
                image_name = osp.join(train_image_dir, f"{epoch}_{iteration}.png")
                save_image(outputs[-1][0].detach().cpu().numpy(), gt[0].detach().cpu().numpy(), image_name)

        if rank == 0:
            avg_loss = epoch_loss / max(iter_num, 1)
            logger.info("epoch: {}, avg_loss: {:.5f}, time: {:.2f}s.".format(epoch, avg_loss, time.time() - start_time))

        if rank == 0 and epoch % cfg.checkpoint_config.interval == 0:
            save_model = model.module if args.distributed else model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": save_model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, osp.join(checkpoints_dir, f"epoch_{epoch}.pth"))


if __name__ == "__main__":
    main()
