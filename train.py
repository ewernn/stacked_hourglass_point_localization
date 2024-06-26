import os
from os.path import dirname
import tqdm
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import importlib
import argparse
from datetime import datetime
from pytz import timezone
import shutil
import wandb
import copy

from data.VHS.vhs_loader import CoordinateDataset
from torch.utils.data import DataLoader
from models.layers import Hourglass


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue_exp', type=str, help='continue exp')
    parser.add_argument('-e', '--exp', type=str, default='eq_neck', help='experiments name')
    parser.add_argument('-m', '--max_iters', type=int, default=250, help='max number of iterations (thousands)')
    parser.add_argument('-p', '--pretrained_model', type=str, help='path to pretrained model')
    parser.add_argument('-o', '--only10', type=bool, default=False, help='only use 10 images')
    parser.add_argument('-w', '--use_wandb', type=bool, default=False, help='log in wandb')
    parser.add_argument('-s', '--no_sweep', action='store_true', help='track run without using sweep_config')
    args = parser.parse_args()
    return args

sweep_config = {
    'method': 'random',  # Can be 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'loss',
        'goal': 'minimize'   
    },
    'parameters': {
        'learning_rate': {
            'min': 0.00005,
            'max': 0.001
        },
        'batch_size': {
            'values': [4]
        },
        # Add other hyperparameters here
    }
}

    
def reload(config):
    """
    Load model's parameters and optimizer state from a checkpoint.
    """
    opt = config['opt']
    
    if opt['pretrained_model'] is not None:
        resume_file = opt['pretrained_model']

        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            
            config['inference']['net'].load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                config['train']['optimizer'].load_state_dict(checkpoint['optimizer'])
            if 'epoch' in checkpoint:
                config['train']['epoch'] = checkpoint['epoch']
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(resume_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_file))

def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    """
    from pytorch/examples
    """
    basename = dirname(filename)
    if not os.path.exists(basename):
        os.makedirs(basename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pt')

def save(config):
    config['inference']['net'].eval()

    resume = '/content/drive/MyDrive/MM/EqNeck/exps2'

    if config['opt']['continue_exp'] is not None:
        resume = os.path.join(resume, config['opt']['continue_exp'])
    else:
        resume = os.path.join(resume, config['opt']['exp'])
    lr_, bs_, = config['train']['learning_rate'], config['train']['batch_size']
    resume_file = os.path.join(resume, f'checkpoint_{lr_}_{bs_}.pt')
    
    save_checkpoint({
            'state_dict': config['inference']['net'].state_dict(),
            'optimizer' : config['train']['optimizer'].state_dict(),
            'epoch': config['train']['epoch'],
        }, False, filename=resume_file)
    print(f'=> save checkpoint at {resume_file}')

def train(train_func, config, post_epoch=None):
    print("Training configuration:")
    print(config['train'])
    
    batch_size = config['train']['batch_size']
    heatmap_res = config['train']['output_res']
    im_sz = config['inference']['inp_dim']
    
    print(f"Batch size: {batch_size}, Heatmap resolution: {heatmap_res}, Image size: {im_sz}")

    data_dir = '/content/drive/MyDrive/MM/EqNeck/EqNeckImages/'

    train_dataset = CoordinateDataset(root_dir=data_dir, csv='Train', im_sz=im_sz,\
            output_res=heatmap_res, augment=True, only10=config['opt']['only10'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    valid_dataset = CoordinateDataset(root_dir=data_dir, csv='Test', im_sz=im_sz,\
            output_res=heatmap_res, augment=False, only10=config['opt']['only10'])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    while True:
        print('epoch: ', config['train']['epoch'])
        if 'epoch_num' in config['train']:
            if config['train']['epoch'] > config['train']['epoch_num']:
                break

        for phase in ['train', 'valid']:
            loader = train_loader if phase == 'train' else valid_loader
            num_step = len(loader)

            print('start', phase, config['opt']['exp'])

            show_range = tqdm.tqdm(range(num_step), total=num_step, ascii=True)
            for i in show_range:
                images, heatmaps = next(iter(loader))
                print(f"Loaded data - Image shape: {images.shape}, Heatmap shape: {heatmaps.shape}")
                datas = {'imgs': images, 'heatmaps': heatmaps}
                train_outputs = train_func(i, config, phase, **datas)

                if 'predictions' in train_outputs:
                    print(f"Model output shape: {[p.shape for p in train_outputs['predictions']]}")

                if config['opt']['use_wandb']:
                    metrics = {
                        "epoch": config['train']['epoch'],
                        "total_loss": train_outputs["total_loss"].item(),
                        "learning_rate": config['train']['learning_rate'],
                        "batch_size": config['train']['batch_size']
                    }
                    prefix = f"{phase}_"
                    wandb.log({prefix + key: value for key, value in metrics.items()})
            
            if phase == 'valid':
                avg_val_loss = sum([train_outputs["total_loss"].item() for _ in show_range]) / len(show_range)

                if avg_val_loss < config['train']['best_val_loss']:
                    config['train']['best_val_loss'] = avg_val_loss
                    print(f"New best validation loss: {avg_val_loss}. Saving model...")
                    save(config)

        config['train']['epoch'] += 1


def init(opt):
    task = importlib.import_module('task.eq_neck')
    config = task.__config__

    opt_dict = vars(opt)

    config['opt'] = opt_dict
    config['train']['epoch'] = 0
    config['train']['best_val_loss'] = float('inf')
    return task, config

def train_with_wandb(task, config):
    config_copy = copy.deepcopy(config)
    config_copy['train']['epoch'] = 0
    wandb.init(config=config_copy)

    config_copy['train']['learning_rate'] = wandb.config.get('learning_rate', config_copy['train']['learning_rate'])
    config_copy['train']['batch_size'] = wandb.config.get('batch_size', config_copy['train']['batch_size'])

    print(f"Updated learning rate: {config_copy['train']['learning_rate']}")
    print(f"Updated batch size: {config_copy['train']['batch_size']}")

    train_func = task.make_network(config_copy)
    reload(config_copy)
    train(train_func, config_copy)
    wandb.finish()


def print_config(config):
    print("=== Configuration ===")
    print(f"Input dimension: {config['inference']['inp_dim']}")
    print(f"Output dimension: {config['inference']['oup_dim']}")
    print(f"Number of stacks: {config['inference']['nstack']}")
    print(f"Input resolution: {config['train']['input_res']}")
    print(f"Output resolution: {config['train']['output_res']}")
    print(f"Batch size: {config['train']['batch_size']}")
    print(f"Learning rate: {config['train']['learning_rate']}")
    print("=====================")

def main():
    opt = parse_command_line()
    task, config = init(opt)

    print_config(config)  # Add this line to print the configuration


    if opt.use_wandb:
        if opt.no_sweep:
            wandb.init(project="eq_1000", config=config)
            train_func = task.make_network(config)
            reload(config)
            train(train_func, config)
            wandb.finish()
        else:
            sweep_id = wandb.sweep(sweep_config, project="eq_1000-hyperparam-sweep")
            wandb.agent(sweep_id, lambda: train_with_wandb(task, config))
    else:
        train_func = task.make_network(config)
        reload(config)
        train(train_func, config)


if __name__ == '__main__':
    main()