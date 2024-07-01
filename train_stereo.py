from __future__ import print_function, division
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from core.srstereov1 import SRStereov1
from core.srstereov1_uncertaintynet import *
from evaluate_stereo import *
import core.srstereov1_stereo_datasets as datasets
import torch.nn.functional as F
try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

from core.utils.utils import bilinear_sampler

import cv2
import random
import pdb
from torchvision.transforms import Resize
from torch.nn.functional import softmax
import time
import shutil
from core.draw import *

class Logger:
    SUM_FREQ = 100
    def __init__(self, model, scheduler, num_steps):
        self.model = model
        self.scheduler = scheduler
        self.num_steps = num_steps
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=args.logdir, filename_suffix=f"-CUDA_{torch.version.cuda}-"+f'torch_{torch.__version__}')
        self.time_total = 0
        self.time_last = time.time()

    def _print_training_status(self):
        # metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        # training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        # metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)

        # eta
        time_now = time.time()
        self.time_total += (time_now - self.time_last)
        self.time_last = time_now
        eta_time = self.time_total/self.total_steps*(self.num_steps-self.total_steps)
        hours, minutes, remaining_seconds = self.convert_seconds(eta_time)
        
        training_str = "[total_steps: {}/{}, eta: {}:{}:{}, lr: {:10.7f}] ".format(self.total_steps+1, self.num_steps, hours, minutes, remaining_seconds, self.scheduler.get_last_lr()[0])
        metrics_data = [k+': '+ str(round(self.running_loss[k]/Logger.SUM_FREQ, 4)) for k in sorted(self.running_loss.keys())]
        metrics_str = (", {}"*len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=args.logdir, filename_suffix=f"-CUDA_{torch.version.cuda}-"+f'torch_{torch.__version__}')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=args.logdir, filename_suffix=f"-CUDA_{torch.version.cuda}-"+f'torch_{torch.__version__}')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()
    
    def convert_seconds(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        return int(hours), int(minutes), int(remaining_seconds)

class Mylogging():
    SUM_FREQ = 100
    def __init__(self, logdir):
        self.lines = [time.strftime(os.path.basename(logdir) + '_%Y-%m-%d %H:%M:%S',time.localtime(time.time())), f"\n",
                          f"CUDA version: {torch.version.cuda}", f"\n",
                          f'torch version: {torch.__version__}', f"\n"]
        self.mylogging_path = os.path.join(logdir, time.strftime('result_%Y-%m-%d_%H:%M:%S-',time.localtime(time.time()))
                                           +f"CUDA_{torch.version.cuda}-"
                                           +f'torch_{torch.__version__}'+'.log') 
        self.total_steps = 0
        self.running_loss = {}

    def add(self, line:str):
        self.lines.append(f"{line}\n")

    def _print_training_status(self):
        self.add(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        for k in self.running_loss:
            self.add("{} {} {}".format(self.total_steps, k, self.running_loss[k]/Mylogging.SUM_FREQ))
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Mylogging.SUM_FREQ == Mylogging.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        for key in results:
            self.add("{} {} {}".format(self.total_steps, key, results[key]))

    def get_log(self):
        return self.lines
    
    def save(self):
        n = open(self.mylogging_path,'w')
        n.writelines(self.get_log())
        n.close()

def Disp_gt_weight(disp_gt, disp_pred, disp_gt_weight_type, disp_gt_weight_h1):
    disp_gt_weight = torch.clamp(torch.pow((disp_gt - disp_pred).abs(), -disp_gt_weight_h1), 0, 1.5).detach()
    return disp_gt_weight.cuda()

def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, delta_disps, edge_map, args, loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of flow predictions """
    # disp_pred:[torch.Size([4, 1, 320, 736])]*22
    # disp_init_pred, disp_gt and valid: torch.Size([4, 1, 320, 736])

    '''
    features_left, features_right: torch.Size([4, 96, 80, 184])
    '''
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    n_predictions = len(disp_preds)
    assert n_predictions >= 1

    disp_loss = 0.0 
    
    # DAPE
    edge_loss = torch.tensor(0.0).to(disp_init_pred.device)
    if args.edge_estimator: # if train the edge_estimator
        kernel_x = [[-1.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(disp_gt.device)
        kernel_y = [[-1.0, -1.0, -1.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(disp_gt.device)

        edge_map_dx_gt = F.conv2d(disp_gt, kernel_x, padding=1) 
        edge_map_dy_gt = F.conv2d(disp_gt, kernel_y, padding=1)
        edge_map_gt = torch.abs(edge_map_dx_gt + edge_map_dy_gt)

        edge_map_gt[:,:,:,0] = 0
        edge_map_gt[:,:,:,-1] = 0
        edge_map_gt[:,:,0,:] = 0
        edge_map_gt[:,:,-1,:] = 0

        edge_map_gt = F.sigmoid(10*(edge_map_gt - 5))
        edge_map_gt = edge_map_gt.detach()
        edge_loss = F.smooth_l1_loss(edge_map[valid.bool()], edge_map_gt[valid.bool()], size_average=True)

    elif args.edge_supervised: # if fine-tune the model with DAPE
        kernel_x = [[-1.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(disp_preds[-1].device)
        kernel_y = [[-1.0, -1.0, -1.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(disp_preds[-1].device)

        for idx_disp_pred, disp_pred in enumerate([disp_init_pred] + disp_preds):

            edge_map_dx_gt = F.conv2d(disp_pred, kernel_x, padding=1) 
            edge_map_dy_gt = F.conv2d(disp_pred, kernel_y, padding=1)
            edge_map_gt = torch.abs(edge_map_dx_gt + edge_map_dy_gt)

            edge_map_gt[:,:,:,0] = 0
            edge_map_gt[:,:,:,-1] = 0
            edge_map_gt[:,:,0,:] = 0
            edge_map_gt[:,:,-1,:] = 0

            edge_map_gt = F.sigmoid(10*(edge_map_gt - 5))
            edge_map = edge_map.detach()
            
            adjusted_loss_gamma = loss_gamma**(15/(n_predictions))
            i_weight = adjusted_loss_gamma**(n_predictions - idx_disp_pred)
            valid_bg = edge_map <= args.edge_supervised_thr
            edge_loss += i_weight * F.smooth_l1_loss(edge_map[valid_bg], edge_map_gt[valid_bg], size_average=True) 

    disp_loss += edge_loss

    init_weight = 1.0
    if args.disp_gt_weight:
        disp_gt_weight = Disp_gt_weight(disp_gt, disp_init_pred, args.disp_gt_weight_type, args.disp_gt_weight_h1)
        init_weight *= disp_gt_weight

    init_disp_loss = 1.0 * F.smooth_l1_loss((init_weight * disp_init_pred)[valid.bool()], (init_weight * disp_gt)[valid.bool()], size_average=True)
    disp_loss += init_disp_loss

    stepwise_loss = torch.tensor(0.0).to(disp_gt.device)
    if args.stepwise and 'kitti' not in args.train_datasets and not args.edge_supervised:
        n_step = len(delta_disps)
        b, _, h, w = delta_disps[0][0].shape
        torch_resize = Resize([h, w])
        disp_gt_down_sample = torch_resize(disp_gt) / (disp_gt.shape[3] / w) # torch.Size([4, 1, 80, 184])
        valid_down_sample = torch_resize(valid) # torch.Size([4, 1, 80, 184])
        valid_down_sample = (valid_down_sample >= 0.5)

        for i, (delta_disp, disp) in enumerate(delta_disps):

            i_loss_weight = 1.0
            adjusted_loss_gamma = loss_gamma**(15/(n_step - 1))
            i_weight = adjusted_loss_gamma**(n_step - i - 1)
            disp_gt_down_sample_now = torch.clamp(disp_gt_down_sample - disp, -args.xga_uncertain_aft_m*1.5, args.xga_uncertain_aft_m*1.5)
            stepwise_loss += i_weight * F.smooth_l1_loss((i_loss_weight * delta_disp)[valid_down_sample.bool()], (i_loss_weight * disp_gt_down_sample_now)[valid_down_sample.bool()], size_average=True)

        disp_loss += stepwise_loss

    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        
        i_loss_weight = 1.0

        if args.disp_gt_weight:
            disp_gt_weight = Disp_gt_weight(disp_gt, disp_preds[i], args.disp_gt_weight_type, args.disp_gt_weight_h1)
            i_loss_weight *= disp_gt_weight

        # SR-Stereo_v1
        if args.stepwise:
            if i == 0:
                disp_previous = disp_init_pred
            else:
                disp_previous = disp_preds[i-1]

            disp_gt_now = disp_previous + torch.clamp(disp_gt - disp_previous, -args.xga_uncertain_aft_m*6, args.xga_uncertain_aft_m*6) 
                
            i_loss = (i_loss_weight * (disp_preds[i] - disp_gt_now.detach())).abs()
        else:
            i_loss = (i_loss_weight * (disp_preds[i] - disp_gt)).abs()

        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return disp_loss, metrics, init_disp_loss

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler

def train(args):

    model = nn.DataParallel(SRStereov2(args))

    print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    global_batch_num = 0
    batch_pass = 0
    logger = Logger(model, scheduler, args.num_steps)
    mylogging = Mylogging(args.logdir)
    
    model.cuda()
    model.train()
    model.module.freeze_bn() # We keep BatchNorm frozen

    validation_frequency =  int(args.num_steps/20) 
    validation_all_frequency = int(args.num_steps/20)
    save_checkpoint_frequency = int(args.num_steps/2) 

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    while should_keep_training:

        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            
            if args.resume_pth is not None and batch_pass != 0:
                batch_pass -= 1
                continue
    
            optimizer.zero_grad()
            
            if 'kitti_sample' in args.train_datasets and args.edge_supervised:
                image1, image2, disp_gt, valid, edge_dape = [x.cuda() for x in data_blob]
            elif 'kitti' in args.train_datasets and args.edge_supervised:
                image1, image2, disp_gt, valid, edge_dape = [x.cuda() for x in data_blob]
            elif 'eth3d_srstereo' in args.train_datasets and args.edge_supervised:
                image1, image2, disp_gt, valid, edge_dape = [x.cuda() for x in data_blob]
            elif ('middlebury_srstereo_Q' in args.train_datasets or 'middlebury_srstereo_H' in args.train_datasets or 'middlebury_srstereo_F' in args.train_datasets) and args.edge_supervised:
                image1, image2, disp_gt, valid, edge_dape = [x.cuda() for x in data_blob]
            else:
                image1, image2, disp_gt, valid = [x.cuda() for x in data_blob]

            assert model.training
            disp_init_pred, disp_preds, edge_map, delta_disps = model(image1, image2, iters=args.train_iters, disp_gt=disp_gt)
            assert model.training

            if args.edge_estimator:
                edge_input = edge_map
            elif args.edge_supervised:
                edge_input = edge_dape
            else:
                edge_input = None

            loss, metrics, init_disp_loss = sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, delta_disps, edge_input, args, max_disp=args.max_disp, step = (total_steps / args.num_steps))
            
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            logger.writer.add_scalar(f'init_disp loss', init_disp_loss.item(), global_batch_num)

            mylogging.add("{} live_loss {}".format(global_batch_num, loss.item()))
            mylogging.add("{} learning_rate {}".format(global_batch_num, optimizer.param_groups[0]['lr']))
            mylogging.add("{} init_disp loss {}".format(global_batch_num, init_disp_loss.item()))

            global_batch_num += 1
            scaler.scale(loss).backward()


            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            logger.push(metrics)
            mylogging.push(metrics)
            
            if total_steps % validation_frequency == validation_frequency - 1: 
                results_1 = validate_sceneflow_400(model.module, iters=args.valid_iters)
                results_2 = validate_middlebury(model.module, iters=args.valid_iters, split='H')
                results_3 = validate_middlebury(model.module, iters=args.valid_iters, split='Q')
                results_4 = validate_eth3d(model.module, iters=args.valid_iters)
                results_5 = validate_kitti(model.module, iters=args.valid_iters)
                results = {}
                results.update(results_1)
                results.update(results_2)
                results.update(results_3)
                results.update(results_4)
                results.update(results_5)

                logger.write_dict(results)
                mylogging.write_dict(results)
                model.train()
                model.module.freeze_bn()

            if total_steps % (save_checkpoint_frequency//2) == (save_checkpoint_frequency//2) - 1:

                save_path = Path(args.logdir + '/%d_%s.pth' % (total_steps + 1, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)
                save_last_path = Path(args.logdir + '/%s_last.pth' % (args.name))
                torch.save(model.state_dict(), save_last_path)

                resume_checkpoint = {
                    'optimizer': optimizer.state_dict(),
                    'lr_schedule': scheduler.state_dict(),
                    'last_steps': total_steps+1,
                    'global_batch_num': global_batch_num,
                    'batch_pass': i_batch+1,
                    'time_total': logger.time_total,
                    'running_loss': logger.running_loss
                    }
                
                torch.save(resume_checkpoint, Path(args.logdir + '/%s_last_resume.pth' % (args.name)))

            total_steps += 1
            mylogging.save()

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 10000:
            save_path = Path(args.logdir + '/%d_epoch_%s.pth.gz' % (total_steps + 1, args.name))
            logging.info(f"Saving file {save_path}")
            torch.save(model.state_dict(), save_path)

    print("FINISHED TRAINING")
    logger.close()
    PATH = args.logdir + '/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='sr-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help="load the weights from a specific checkpoint")
    parser.add_argument('--resume_pth', default=None, help="resume the training from a specific checkpoint")
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--logdir', default='./checkpoints/sceneflow', help='the directory to save logs and checkpoints')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.") #200000
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 736], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=22, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # srstereo
    parser.add_argument('--stepwise', action='store_true', help='use stepwise')
    parser.add_argument('--xga_uncertain_aft', action='store_true', help='use xga_uncertain')
    parser.add_argument('--xga_uncertain_aft_type', type=int, default=6, help='xga_uncertain_type')
    parser.add_argument('--xga_uncertain_aft_m', type=float, default=2, help='xga_uncertain_aft_m')
    parser.add_argument('--disp_gt_weight', action='store_true', help='use disp_gt_weight')
    parser.add_argument('--disp_gt_weight_type', type=int, default=9, help='disp_gt_weight_type')
    parser.add_argument('--disp_gt_weight_h1', type=float, default=0.5, help='disp_gt_weight_h1')

    # DAPE
    parser.add_argument('--edge_estimator', action='store_true', help='use edge_estimator')
    parser.add_argument('--edge_supervised', action='store_true', help='use edge_supervised')
    parser.add_argument('--edge_supervised_thr', type=float, default=0.25, help='edge_supervised_thr')

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently") #不调用则为false
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification') #不调用则为false
    args = parser.parse_args()

    torch.manual_seed(666)
    np.random.seed(666)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path(args.logdir).mkdir(exist_ok=True, parents=True)

    train(args)
