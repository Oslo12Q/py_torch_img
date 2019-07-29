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


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, dataset, my_loader, selected_attrs,imagename, model_save_dir,result_dir):
        """Initialize configurations."""
        self.imagename = imagename

        # Data loader.

        self.my_loader = my_loader

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
        self.use_tensorboard = False
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
        if self.use_tensorboard:
            self.build_tensorboard()

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

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    

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
                print(len(c_trg_list))
                # Translate images.
                x_fake_list = []
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))
                    # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}.jpg'.format(self.imagename))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)


   

