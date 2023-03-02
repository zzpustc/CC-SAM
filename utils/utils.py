#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torchvision import transforms

import os
from os import path as osp

import logging
import numpy as np
from datetime import datetime

from utils.lr_scheduler import WarmupMultiStepLR

# Data transformation with augmentation
data_transforms_inat = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
    ])
}

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.module.classifier.parameters():
        param.requires_grad = True

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if isinstance(v, dict):
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

def transform_selection(cfg, mode):
    if cfg['dataset']['dataset_name'] == 'iNat2018':
        return data_transforms_inat[mode]
    else:
        return data_transforms[mode]

def pre_compute_class_ratio(cfg, data_source):
    ratios = []
    labels = data_source.labels
    num_classes = len(np.unique(labels))
    cls_data_list = [list() for _ in range(num_classes)]
    for i, label in enumerate(labels):
        cls_data_list[label].append(i)
    max_num = 0
    for cls in cls_data_list:
        tmp = len(cls)
        ratios.append(tmp)
        if tmp > max_num:
            max_num = tmp
    num_per_class = ratios
    weights = np.log(max_num / np.array(ratios) + 0.01) / cfg['train']['div']     # to prevent zero 
    ratios = np.array(ratios) #/ max_num + 1.0e-5 #cfg['train']['tolerant']     # to prevent zero 

    return num_per_class, ratios, weights

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def create_logger(cfg, rank=0, test=False):
    dataset = cfg['dataset']['dataset_name']
    if cfg['debug']:
        dataset = "debug"
    backbone_name = cfg['backbone']['name']
    head_type = cfg['head']['type']
    if test: # for testing
        log_dir = osp.join(cfg['output_dir'], dataset, "test")
        log_name = '{}.log'.format(cfg['test']['exp_id'])
        log_file = osp.join(log_dir, log_name)
    else:
        log_dir = osp.join(cfg['output_dir'], dataset, "logs")
        time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        # log_name = "{}_{}_{}_{}_{}.log".format(dataset, drug_encoding, protein_encoding, head_type, time_str)

        loss = cfg['loss']['type']
        seed = cfg['seed']
        log_name = "{}_{}_{}_{}_{}_{}.log".format(dataset, backbone_name, loss, seed, head_type, time_str)

        log_file = osp.join(log_dir, log_name)
    if not osp.exists(log_dir) and rank == 0:
        os.makedirs(log_dir)

    # set up logger
    print("=> creating log {}".format(log_file))
    header = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=header)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if rank > 0:
        return logger, log_file
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Cfg is set as follow--------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file, log_name.split('.')[0]


# load the pre-trained model
def init_weights(model, weights_path, caffe=False, classifier=False):  
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))    
    weights = torch.load(weights_path)
    if not classifier:
        if caffe:
            weights = {k: weights[k] if k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
    else:      
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k] 
                   for k in model.state_dict()}
    model.load_state_dict(weights)   
    return model

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20):
    
    training_labels = np.array(train_data.labels).astype(int)
#    preds = preds.argmax(dim=1)
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    overall_shot = []
    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        overall_shot.append((class_correct[i] / test_class_count[i]))
        if train_class_count[i] >= many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] <= low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))          
    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), np.mean(overall_shot)

def get_optimizer(cfg, model):

    optim_type = cfg['train']['optimizer']['type']
    params = []

    cls_params = list(map(id, model.module.classifier.parameters())) #+ list(map(id, model.module.fc_add.parameters()))
    base_params = filter(lambda p: id(p) not in cls_params, model.parameters())
    params = [{'params': base_params, 'lr': cfg['train']['optimizer']['lr']},
            {'params': model.module.classifier.parameters(), 'lr': cfg['train']['optimizer']['lr_cls']}
      ]

    if optim_type == "SGD":
        optimizer = torch.optim.SGD(
            params=params,
            lr=cfg['train']['optimizer']['lr_neck'],
            momentum=cfg['train']['optimizer']['momentum'],
            weight_decay=cfg['train']['optimizer']['wc'],
         #   nesterov=True,
        )
    elif optim_type == "ADAM":
        optimizer = torch.optim.Adam(
            params=params,
            lr=cfg['train']['optimizer']['lr_neck'],
            betas=(0.9, 0.999),
            weight_decay=cfg['train']['optimizer']['wc'],
        )
    else:
        raise NotImplementedError
    return optimizer

def get_scheduler(cfg, optimizer, t_max):
    if cfg['train']['lr_scheduler']['type'] == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=cfg['train']['lr_scheduler']['lr_step'],
            gamma=cfg['train']['lr_scheduler']['lr_factor'],
        )
    elif cfg['train']['lr_scheduler']['type'] == "cosine":
        if cfg['train']['lr_scheduler']['cosine_decay_end'] > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=cfg['train']['lr_scheduler']['cosine_decay_end'],
                eta_min=1e-4,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max= t_max,
                eta_min= 0,
            )
    elif cfg['train']['lr_scheduler']['type'] == "warmup":
        scheduler = WarmupMultiStepLR(
            optimizer=optimizer,
            milestones=cfg['train']['lr_scheduler']['lr_step'],
            gamma=cfg['train']['lr_scheduler']['lr_factor'],
            warmup_epochs=cfg['train']['lr_scheduler']['warmup_epoch'],
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(cfg['train']['lr_scheduler']['type']))

    return scheduler


def reset_weight(model, pretrained_path):
    
    pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model
    
def lr_reset(cfg, model):
    if 'Cifar' in cfg['dataset']['dataset_name']:
        lr_new = cfg['train']['optimizer']['lr_neck']
    else:
        lr_new = cfg['train']['optimizer']['lr_neck'] * cfg['train']['lr_scheduler']['lr_factor']
    optimizer = torch.optim.SGD(
            params= model.module.classifier.parameters(),
            lr= lr_new,
            momentum=cfg['train']['optimizer']['momentum'],
            weight_decay=cfg['train']['optimizer']['wc'],
        )
        
    return optimizer

def norm_clip(noise, noise_norm = 1.0e-3):
    abnorm = torch.norm(noise)
    if abnorm <= noise_norm:
        return noise
    
    else:
        vec_noise = noise  / abnorm 
        return vec_noise * noise_norm

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def param_count(model):

    params = list(model.parameters())
    k=0
    for i in params:
        l=1
        # print("This layer：" + str(list(i.size())))
        for j in i.size():
            l*=j
        # print("Params of this layer：" + str(l))
        k = k + l
    # print("Total params：" + str(k))

    return k