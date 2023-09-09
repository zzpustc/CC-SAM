import torch
import os
import random
import numpy as np
import torch.nn as nn

from torch.nn.parallel import DataParallel
import torch
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply

"""
GPU wrappers
"""

_use_gpu = False
device = None

def grad_norm(param_groups, device):
    shared_device = device  # put everything on the same device, in case of model parallelism
    p_norm = torch.norm(
        torch.stack([
            p.norm(p=2).to(shared_device) for p in param_groups
        ]
        ),
        p=2
    )
    return p_norm 

def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

class BalancedDataParallel(DataParallel):
    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids)
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]
        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    if _use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_id)
    return torch.device("cuda:0" if _use_gpu else "cpu")

def global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) # gpu
    torch.cuda.manual_seed_all(seed) # gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def gpu_enabled():
    return _use_gpu

def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return from_numpy(elem_or_tuple).float()


def filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    if isinstance(np_batch, dict):
        return {
            k: elem_or_tuple_to_variable(x)
            for k, x in filter_batch(np_batch)
            if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
        }
    else:
        return from_numpy(np_batch)

def init_weight(m, initrange=0.1, zero_bias=False):
    if hasattr(m, 'weight'):
        m.weight.data.uniform_(-initrange, initrange)
        if hasattr(m, 'bias') and zero_bias:
            m.bias.data.zero_()

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def move_to_device(obj, device=None):
    if (device is None):
        device = torch.device('cuda')
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        return obj.to(device)

# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    return torch.FloatTensor(*args, **kwargs).to(device)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    # not sure if I should do detach or not here
    return tensor.to('cpu').detach().numpy()


def zeros(*sizes, **kwargs):
    return torch.zeros(*sizes, **kwargs).to(device)


def ones(*sizes, **kwargs):
    return torch.ones(*sizes, **kwargs).to(device)


def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs).to(device)


def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs).to(device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)

def initialize_weights(model: nn.Module):
    """
    Initializes the weights of a model in place.
    :param model: An nn.Module.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
