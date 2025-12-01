import torch
import datetime
import os
import logging

import omegaconf

from utils.logger import prepare_logging
from torch.utils.data import DataLoader
from data.datasets import get_datasets
from models.model import NamedCurves
from utils.setup_optim_scheduler import get_optimizer_scheduler
from utils.evaluator import Evaluator
from utils.setup_criterion import get_criterion

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def main(config: omegaconf.DictConfig):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.train.cuda_visible_device)
    # os.environ["CUDA_VISIBLE_DEVICES"]是设置当前Python进程能看到哪些GPU的环境变量。
    # 如果os.environ["CUDA_VISIBLE_DEVICES"]="2,3"，那么程序里torch.cuda.device_count()会返回2
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    is_main = (rank == 0)
    if is_main:
        print(f'[Main] world_size={world_size} backend = "nccl" ')

    save_path = prepare_logging()
    if is_main:
        logging.info(f"Starting training at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Saving logs to {save_path}")
        logging.info(f"Config file: {OmegaConf.to_yaml(config)}")

    train_dataset, valid_dataset, test_dataset = get_datasets(config.data)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    if is_main:
        msg = f"Training with {len(train_dataset)} image pairs"
        if valid_dataset is not None:
            msg += f", validating with {len(valid_dataset)} image pairs"
        msg += f" and testing with {len(test_dataset)} image pairs."
        logging.info(msg)

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, sampler=train_sampler, num_workers=os.cpu_count(), pin_memory=True)
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    else:
        valid_loader = None
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = NamedCurves(config.model).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    if config.model.ckpt_path is not None:
        model.module.load_state_dict(torch.load(config.model.ckpt_path)["model_state_dict"])  # model.load_state_dict参数是字典

    criterion = get_criterion(config.train.criterion)

    optimizer, scheduler = get_optimizer_scheduler(model,
                                                   config.train.optimizer,
                                                   config.train.scheduler if "scheduler" in config.train else None)

    if valid_loader is not None:
        valid_evaluator = Evaluator(valid_loader, config.eval.metrics, 'valid', save_path, config.eval.metric_to_save)
    else:
        valid_evaluator = None

    for epoch in range(config.train.epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
        model.train()
        for data in train_loader:
            input_image, target_image, name = data['input_image'], data['target_image'], data['name']
            optimizer.zero_grad()
            prediction, x_backbone = model(input_image.to(device), return_backbone = True)
            loss = criterion(x_backbone, prediction, target_image.to(device))
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(train_loader)
        if is_main:
            logging.info(f"Epoch {epoch+1}/{config.train.epochs} | Loss: {epoch_loss}")
        if valid_loader is not None and (epoch+1) % config.train.valid_every == 0:
            valid_evaluator(model)

    if is_main:
        test_evaluator = Evaluator(test_loader, config.eval.metrics, 'test', save_path, config.eval.metric_to_save)
        test_evaluator(model, save_results=True if valid_evaluator is None else False)
        logging.info("Training finished.")
    dist.destroy_process_group()

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mit5k_dpe_config.yaml')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main(config)
