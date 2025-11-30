import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os.path
from datetime import datetime
import json, random, datetime, wandb
from argparse import ArgumentParser
from pathlib import Path
import multiprocessing 
from torch.cuda.amp import autocast

import numpy as np
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist

from statistics import mean, geometric_mean, harmonic_mean
from collections import OrderedDict

from trainer import Trainer
from utils import *
from data.lasco_dataset import LaSCoDataset
from data.igrot_dataset import IGROTDataset
from data.cirr_dataset import CIRRDataset
from data.fiq_dataset import FashionIQDataset
from data.sketchy_dataset import SketchyDataset
from data_submission.circo_test_submission import main_circo 
from data_submission.cirr_test_submission import main_cirr 
from data_submission.fiq_test_submission import main_fiq
from data_submission.dtin_test_submission import main_dtin
from data_submission.sketchy_test_submission import main_sketchy
from data_submission.tuberlin_test_submission import main_tuberlin
from data_submission.quickdraw_test_submission import main_quickdraw
from data_submission.patterncom_test_submission import main_patterncom
from data_submission.pku_test_submission import main_pku
from data_submission.igrot_test_submission import main_igrot
import comet_ml

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def main(cfg):

    setup_seed(0)
    # Set up model 

    model = get_model(cfg)
    set_grad(cfg, model)
    model.model.eval().float()
    if cfg.model.startswith('blip'):
        input_dim = 384
    elif cfg.model.startswith('clip'):
        input_dim = model.model.visual.input_resolution
    elif cfg.model.startswith('coca'):
        input_dim = 224
    elif cfg.model.startswith('siglip'):
        input_dim = 256
    preprocess = get_preprocess(cfg.preprocess, input_dim)
    llava_caption = True if cfg.llava == 'with' else False
    data_amount = cfg.data_amount
    print('Llava: ', llava_caption)
    if cfg.dataset == 'lasco':
        relative_train_dataset = LaSCoDataset(
            split = 'train', 
            mode = 'relative',
            preprocess = preprocess,
            llava = llava_caption)
    elif cfg.dataset == 'sketchy':
        relative_train_dataset = SketchyDataset(
                split = 'train',
                domain_type = ['0'],
                mode = 'relative',
                preprocess = preprocess)
    elif cfg.dataset == 'igrot':
        relative_train_dataset = IGROTDataset(
                split = 'train',
                mode = 'relative',
                preprocess = preprocess, 
                data_amount = data_amount)
    elif cfg.dataset == 'cirr':
        relative_train_dataset = get_laion_cirr_dataset(preprocess, cfg.laion_type)

    #print(relative_train_dataset)
    relative_train_loader = DataLoader(
        dataset = relative_train_dataset,
        batch_size = cfg.batch_size,
        num_workers = multiprocessing.cpu_count(),
        pin_memory = True,
        collate_fn = collate_fn,
        drop_last = True,
        shuffle = True
    )
    # classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    classic_val_dataset = LaSCoDataset('val', 'classic', preprocess, False)
    # val_index_features, val_index_features, val_total_index_features = extract_index_features(classic_val_dataset, model, return_local=False)
    # relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    relative_val_dataset = LaSCoDataset('val', 'relative', preprocess, True)
    
    # Define optimize, loss, and grad scaler 
    optimizer = get_optimizer(model, cfg)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.num_epochs, 
            eta_min=1e-2 * cfg.learning_rate, 
            last_epoch=-1)
    crossentropy_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    trainer = Trainer(
        cfg = cfg, 
        model = model, 
        train_dataloader = relative_train_loader, optimizer = optimizer, scheduler = lr_scheduler, criterion = crossentropy_criterion, classic_val_dataset = classic_val_dataset, relative_val_dataset = relative_val_dataset)
    trainer.train() 


if __name__ == "__main__":
    
    data_path = '/home/hle/Composed_Image_Retrieval/'
    parser = ArgumentParser()
    parser.add_argument("--training", type = str, default = 'True')
    parser.add_argument("--model", type=str, default = "clip_base", help = "['clip_base', 'clip_large', 'blip']")
    parser.add_argument("--encoder", type=str, default = "both", help = "['text', 'both', 'neither']")
    parser.add_argument("--dataset", type=str, default = "lasco")
    parser.add_argument("--laion-type", type=str, default = "laion_combined", help = "['laion_template', 'laion_llm', 'laion_combined']")
    parser.add_argument("--llava", type=str, default = "with", help = "['with', 'without']")
    parser.add_argument("--main_type", type=str, default = "combine", help = "['combine', 'union', 'fuse', 'company']")
    parser.add_argument("--target_type", type=str, default = "original", help = "['original', 'sum', 'union', 'fuse']")
    parser.add_argument("--loss_type", type = str, default = "infonce", help = "['infonce', 'extended_infonce', 'avg_infonce']")
    parser.add_argument("--data_amount", type = int, default = 0)
    parser.add_argument("--batch-size", type=int, default = 32)
    parser.add_argument("--num-epochs", type=int, default = 1)
    parser.add_argument("--num_layers", type = int, default = 2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=1e-1)
    parser.add_argument("--weight_decay", type=float, default = 0.05)
    parser.add_argument("--adam_epsilon", type = float, default = 1e-6)
    parser.add_argument("--validation_frequency", type = int, default = 1)
    parser.add_argument("--preprocess", type = str, default = "targetpad", help = "['squarepad', 'targetpad']")
    parser.add_argument("--vision_projector", type = bool, default = False) 
    parser.add_argument("--comment", type = str)
    parser.add_argument("--save-path", type=str, required=True)
    # Eval related
    parser.add_argument("--inference", type=str, default = 'False')
    parser.add_argument("--val_load_path", type=str)
    parser.add_argument("--val_dataset", type=str)
    parser.add_argument("--submission_name", type=str)
    config = parser.parse_args()
    config.device = torch.device('cuda')
    #pretrained_model = 'ViT-B-32' if 'base' in config.model else 'ViT-L-14'
    #if "ViT" in pretrained_model:
    #    config.model_path = "/home/hle/spinning-storage/hle/ckpt/" + pretrained_model + ".pt"
    #elif "blip" in pretrained_model:
    #    config.model_path = "/home/hle/spinning-storage/hle/ckpt/" + "model_large_retrieval_coco.pth"
    #config.model_type = 'base' if 'base' in config.model else 'large'

    #wandb.login(key = "") 
    
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    config.save_path = f"{config.save_path}.pth"
    if config.training == 'True':   
        if config.dataset == 'sketchy':
            comet = comet_ml.start(project_name=f"sbir_{config.comment}_{config.preprocess}")
        else:
            comet = comet_ml.start(project_name=f"{config.comment}_{config.preprocess}")
        config.comet = comet
    
        #run = wandb.init(
                # Set the project where this run will be logged
        #        project="CIR",
                # Track hyperparameters and run metadata
        #        config={
        #            "learning_rate": config.learning_rate,
        #            "epochs": config.num_epochs,
        #            },
        #        name = config.comment
        #)
        
        main(config)
        #run.log_code()

        #wandb.finish()

    if config.inference == 'True':
        if config.training == 'True':
            config.val_load_path = config.save_path

        inference_dataset = config.val_dataset
        function = globals().get(f"main_{inference_dataset}")
        if function:
            function(config)
        else:
            print(f"No function named main_{func_name} found")
