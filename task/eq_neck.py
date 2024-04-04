"""
__config__ contains the options for training and testing
Basically all of the variables related to training are put in __config__['train'] 
"""
import torch
import numpy as np
from torch import nn
import os
from torch.nn import DataParallel
from utils.misc import make_input, make_output, importNet

import matplotlib.pyplot as plt

__config__ = {
    'data_provider': 'data.MPII.dp',
    'network': 'models.posenet.PoseNet',
    'inference': {
        'nstack': 2,
        'inp_dim': 300,
        'oup_dim': 3,
        'num_parts': 3,
        'increase': 0,
        'keys': ['imgs']
    },
# inference-time compute, gameplay-style value optimization
    'train': {
        'epoch_num': 2,
        'learning_rate': 1e-4,
        'batch_size': 1,
        'input_res': 300,
        'output_res': 75,
        'train_iters': 1000,
        'valid_iters': 10,
        'max_num_people' : 1,
        'loss': [
            ['combined_hm_loss', 1],
        ],
        'decay_iters': 100000,
        'decay_lr': 2e-4,
        'num_workers': 2,
        'use_data_loader': True,
    },
}

class Trainer(nn.Module):
    """
    The wrapper module that will behave differetly for training or testing
    inference_keys specify the inputs for inference
    """
    def __init__(self, model, inference_keys, calc_loss=None):
        super(Trainer, self).__init__()
        self.model = model
        self.keys = inference_keys
        self.calc_loss = calc_loss

    def forward(self, imgs, **inputs):
        inps = {}
        labels = {}

        for i in inputs:
            if i in self.keys:
                inps[i] = inputs[i]
            else:
                labels[i] = inputs[i]
        #print(f"Shape of imgs: {imgs.shape}") ERIC
        if not self.training:
            preds = self.model(imgs, **inps)
            # Assuming you might want to print preds shape during inference too
            #print(f"Shape of preds (inference): {preds.shape if not isinstance(preds, (list, tuple)) else [p.shape for p in preds]}")
            return preds
        else:
            combined_hm_preds = self.model(imgs, **inps)
            if type(combined_hm_preds) != list and type(combined_hm_preds) != tuple:
                combined_hm_preds = [combined_hm_preds]
            #print(f"Shape of combined_hm_preds: {[p.shape for p in combined_hm_preds]}") # ERIC
            true_heatmaps = labels['heatmaps']
            loss = self.calc_loss(combined_hm_preds, true_heatmaps)
            
            # print(f"Shape of true_heatmaps: {true_heatmaps.shape}")
            # print(f"Shape of combined_hm_preds: {combined_hm_preds[0].shape}")
            # print(f"Shape of loss: {loss['combined_total_loss'].shape}")
            # print(f"Shape of output: {len(list(combined_hm_preds) + [loss])}")
            return list(combined_hm_preds) + [loss]
#######################################################################

def make_network(configs):
    train_cfg = configs['train']
    config = configs['inference']

    def calc_loss(*args, **kwargs):
        return poseNet.calc_loss(*args, **kwargs)
        # loss = poseNet.calc_loss(*args, **kwargs)
        # return {"total_loss": loss}
    
    PoseNet = importNet(configs['network'])
    poseNet = PoseNet(**config)
    forward_net = poseNet.cuda()
    if torch.cuda.device_count() > 1:
        forward_net = DataParallel(forward_net)
    config['net'] = Trainer(forward_net, configs['inference']['keys'], calc_loss)

    learning_rate = train_cfg['learning_rate']
    train_cfg['optimizer'] = torch.optim.Adam(filter(lambda p: p.requires_grad, config['net'].parameters()), lr=learning_rate)
    train_cfg['scheduler'] = torch.optim.lr_scheduler.StepLR(train_cfg['optimizer'], step_size=2000, gamma=0.4)
    train_cfg['warmup_steps'] = 0
    train_cfg['initial_lr'] = 1e-3  # This should match the learning rate you set for the optimizer
    train_cfg['warmup_initial_lr'] = 1e-5  # Starting learning rate for warmup
    train_cfg['current_step'] = 0  # To keep track of the current step across batches


    
    ## optimizer, experiment setup
    #exp_path = '/content/drive/MyDrive/MM/EqNeck3pts/exps'
    exp_path = '/home/eawern/Eq/stacked_hourglass_point_localization/exps'
    if configs['opt']['continue_exp'] is not None:  # don't overwrite the original exp I guess ??
        exp_path = os.path.join(exp_path, configs['opt']['continue_exp'])
    else:
        exp_path = os.path.join(exp_path, configs['opt']['exp'])

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    logger = open(os.path.join(exp_path, 'log'), 'a+')

    # **inputs = {'imgs': images, 'heatmaps': heatmaps}
    def make_train(batch_id, config, phase, **inputs):
        for i in inputs:
            try:
                inputs[i] = make_input(inputs[i])
            except:
                pass #for last input, which is a string (id_)
        #print(f"ericcc: \n{inputs['imgs'].shape}")
                
        net = config['inference']['net']
        config['batch_id'] = batch_id

        net = net.train()

        # When in validation phase put batchnorm layers in eval mode
        # to prevent running stats from getting updated.
        if phase == 'valid':
            for module in net.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        #  ERIC: which phase is being used??
        
        
        if phase != 'inference':
            result = net(inputs['imgs'], **{i:inputs[i] for i in inputs if i!='imgs'})
            # Assuming result[0] are the predictions and result[1] are the losses
            #print(f"type of result: {type(result)}")
            combined_hm_preds = result[:-1]
            loss_dict = result[-1]

            combined_total_loss = loss_dict["combined_total_loss"]
            # logger.write(f"c: {combined_total_loss}")
            # logger.flush()

            # Aggregate loss across all stacks
            total_loss = combined_total_loss.mean()

            # Logging
            toprint = f'\n{batch_id}: Total Loss: {total_loss.item():.8f}\n'
            for stack_idx in range(combined_total_loss.shape[1]):
                stack_loss = combined_total_loss[:, stack_idx].mean()
                toprint += f'Stack {stack_idx} Loss: {stack_loss.item():.8f}\n'

            logger.write(toprint)
            logger.flush()

            # Backpropagation and optimization step
            def adjust_learning_rate_during_warmup(optimizer, train_cfg):
                current_step = train_cfg['current_step']
                warmup_steps = train_cfg['warmup_steps']
                if current_step < warmup_steps:
                    # Linear scaling of the learning rate
                    scaled_lr = (train_cfg['initial_lr'] - train_cfg['warmup_initial_lr']) * (current_step / warmup_steps) + train_cfg['warmup_initial_lr']
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = scaled_lr

            # Inside your make_train function, before the optimizer.step() call:
            if phase == 'train':
                optimizer = train_cfg['optimizer']
                adjust_learning_rate_during_warmup(optimizer, train_cfg)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                train_cfg['current_step'] += 1  # Increment the step counter

            # Update the scheduler after warmup is complete
            if train_cfg['current_step'] > train_cfg['warmup_steps']:
                train_cfg['scheduler'].step()


            # Learning rate decay
            if batch_id == config['train']['decay_iters']:
                for param_group in optimizer.param_groups:
                    param_group['learning_rate'] = config['train']['decay_lr']
            return {"total_loss": total_loss,
                    # "basic_loss": basic_loss,
                    # "focused_loss": focused_loss,
                    "predictions": combined_hm_preds}

        else:
            out = {}
            net = net.eval()
            result = net(**inputs)
            print(f"type of result: {type(result)}")
            if type(result)!=list and type(result)!=tuple:
                result = [result]
            out['preds'] = [make_output(i) for i in result]
            return out
    return make_train
