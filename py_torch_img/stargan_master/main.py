import os
import argparse
import cv2
from torchvision import transforms as T
from PIL import Image
from .data_loader1 import  get_loader1
from torch.backends import cudnn
import numpy as np
import torch


def str2bool(v):
    return v.lower() in ('true')

def link(input_dir,imagename,model_save_dir,selected_attrs,result_dir,G):

    log_dir='stargan/logs'
    dataset='Mytest'
    sample_dir='stargan/samples'
    input_dir=input_dir
    imagename=imagename
    model_save_dir=model_save_dir
    selected_attrs=selected_attrs
    result_dir=result_dir
    num_workers=0
    image_size=256
    # For fast training.
    cudnn.benchmark = True
    G=G
   

    # Create directories if not exist.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    orig_img = cv2.imread(os.path.join(input_dir, imagename))
    shape= orig_img.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    # Data loader.

    my_loader = None
    my_loader = get_loader1 (input_dir, imagename, dataset, image_size, num_workers)

    # Solver for training and testing StarGAN.
    test(my_loader,device,selected_attrs,result_dir,shape,imagename,G)
    
def denorm(x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

def create_labels_test(device,selected_attrs):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        for i, attr_name in enumerate(selected_attrs):
            attrs_list=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
            index = attrs_list.index(attr_name)
        c_trg_list = []
        data=np.array([[0, 0, 1, 1, 1]], dtype=np.float32)
        c_trg = torch.from_numpy(data)
        c_trg[:, index] = 1
        c_trg_list.append(c_trg.to(device))
        return c_trg_list

def test(my_loader,device,selected_attrs,result_dir,shape,imagename,G):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.


        # Set data loader.
 
         
        with torch.no_grad():
            for i, (x_real) in enumerate(my_loader):
                # Prepare input images and target domain labels.
                x_real = x_real.to(device)
                c_trg_list =create_labels_test(device, selected_attrs)
                # Translate images.
                c_trg = c_trg_list[0]

                x_fake =G(x_real, c_trg)
                # Save the translated images.
                x_result = denorm(x_fake.data.cpu())
                result_path = os.path.join(result_dir, '{}.jpg'.format(imagename))

                result_new = T.ToPILImage()(torch.squeeze(x_result)).convert('RGB')
                result_new = result_new.resize((shape[1],shape[0]),Image.BILINEAR)
                result_new.save(result_path)

