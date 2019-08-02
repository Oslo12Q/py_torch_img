from .model import Generator
from .model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import cv2

# 导入model 路径
from django.conf import settings

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self,model_save_dir):
        """Initialize configurations."""
        # Data loader.

     

       # Model configurations.
        self.c_dim = 5
        self.c2_dim = 8
        self.image_size = 256
        self.g_conv_dim = 64
        self.d_conv_dim = 64
        self.g_repeat_num = 6
        self.d_repeat_num = 6
        self.lambda_cls = 1.0
        self.lambda_rec = 10.0
        self.lambda_gp = 10.0

        # Training configurations.
    
        self.batch_size = 1
        self.num_iters = 200000
        self.num_iters_decay = 100000
        self.g_lr = 0.0001
        self.d_lr = 0.0001
        self.n_critic = 5
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.resume_iters =None
   

        # Test configurations.
        self.test_iters = 200000

        # Miscellaneous.


        #self.device = torch.device('cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = 'stargan/logs'
        self.sample_dir = 'stargan/samples'
        self.model_save_dir = model_save_dir
  

        # Step size.
        self.log_step = 10
        self.sample_step = 1000
        self.model_save_step = 10000
        self.lr_update_step = 1000
        self.resume_iters=200000
        # Build the model and tensorboard.

        self.build_model()     
        self.restore_model(self.resume_iters)

    def build_model(self):
        """Create a generator and a discriminator."""    
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.G.to(self.device)
        self.D.to(self.device)



    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
#        self.G.load_state_dict(torch.load(G_path))
#        self.D.load_state_dict(torch.load(D_path))
        
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
solver=Solver(settings.MODEL_PATH)
