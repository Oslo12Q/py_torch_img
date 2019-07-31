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
from torchvision import transforms as T
from PIL import Image

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, dataset, my_loader, selected_attrs,imagename, model_save_dir,result_dir,shape):
        """Initialize configurations."""
        self.imagename = imagename

        # Data loader.

        self.my_loader = my_loader
        self.shape=shape

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
        self.dataset = dataset
        self.batch_size = 1
        self.num_iters = 200000
        self.num_iters_decay = 100000
        self.g_lr = 0.0001
        self.d_lr = 0.0001
        self.n_critic = 5
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.resume_iters =None
        self.selected_attrs =selected_attrs

        # Test configurations.
        self.test_iters = 200000

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = 'stargan/logs'
        self.sample_dir = 'stargan/samples'
        self.model_save_dir = model_save_dir
        self.result_dir =result_dir

        # Step size.
        self.log_step = 10
        self.sample_step = 1000
        self.model_save_step = 10000
        self.lr_update_step = 1000

        # Build the model and tensorboard.
        self.build_model()


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
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

  

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def create_labels_test(self, selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        for i, attr_name in enumerate(selected_attrs):
            attrs_list=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
            index = attrs_list.index(attr_name)
        c_trg_list = []
        data=np.array([[0, 0, 1, 1, 1]], dtype=np.float32)
        c_trg = torch.from_numpy(data)
        c_trg[:, index] = 1
        c_trg_list.append(c_trg.to(self.device))
        return c_trg_list



  
    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
 
        data_loader = self.my_loader
        with torch.no_grad():
            for i, (x_real) in enumerate(data_loader):
                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels_test(self.selected_attrs)
                # Translate images.
                c_trg = c_trg_list[0]

                x_fake = self.G(x_real, c_trg)
                # Save the translated images.
                x_result = self.denorm(x_fake.data.cpu())
                result_path = os.path.join(self.result_dir, '{}-out.jpg'.format(self.imagename))

                result_new = T.ToPILImage()(torch.squeeze(x_result)).convert('RGB')
                result_new = result_new.resize((self.shape[1],self.shape[0]),Image.BILINEAR)
                result_new.save(result_path)
                

    
                


   

