import os
import argparse
from .solver import Solver
from .data_loader1 import  get_loader1
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def link(input_dir,imagename,model_save_dir,selected_attrs,result_dir):

    log_dir='stargan/logs'
    dataset='Mytest'
    sample_dir='stargan/samples'
    input_dir=input_dir
    imagename=imagename
    model_save_dir=model_save_dir
    selected_attrs=selected_attrs
    result_dir=result_dir
    num_workers=1
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Data loader.

    my_loader = None
    my_loader = get_loader1 (input_dir, imagename, dataset, num_workers)

    # Solver for training and testing StarGAN.
   
    solver = Solver(dataset, my_loader, selected_attrs,imagename, model_save_dir,result_dir)
    solver.test()


