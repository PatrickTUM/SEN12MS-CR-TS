"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init


def tensor2im(input_image, method, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # no need to do anything if image_numpy is 3-dimensiona already but for the other dimensions ...
        
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1)) # triple channel
            image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
        
        if image_numpy.shape[0] == 13 or image_numpy.shape[0] == 4: # 13 bands multispectral (or 4 bands NIR) to RGB
            # RGB bands are [3, 2, 1]
            image_numpy = image_numpy[[3, 2, 1], ...]   
            
            # method is either 'resnet' (if opt.alter_initial_mode) or 'default'
            if method == 'default': # re-normalize from [-1,+1] to [0,+1]
                image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            elif method == 'resnet': # re-normalize from [0, 5] to [0,+1]
                image_numpy = (np.transpose(image_numpy, (1, 2, 0))) / 5.0 * 255.0
        
        if image_numpy.shape[0] == 2:  # (VV,VH) SAR to RGB (just taking VV band)
            image_numpy = np.tile(image_numpy[[0]], (3, 1, 1))
            if method == 'default': # re-normalize from [-1,+1] to [0,+1]
                image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            elif method == 'resnet':  # re-normalize from [0, 2] to [0,+1]
                image_numpy = (np.transpose(image_numpy, (1, 2, 0))) / 2.0 * 255.0
          # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def weights_init_kaiming(m):  # initialize the weights (kaiming method)
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)


def fc_init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)


class VGG16(nn.Module):
    def __init__(self, n_inputs=12, numCls=17):  # num. of classes
        super().__init__()

        vgg = models.vgg16(pretrained=False)

        self.encoder = nn.Sequential(
            nn.Conv2d(n_inputs, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 12 bands as input (s1 + s2)
            *vgg.features[1:]
        )
        self.classifier = nn.Sequential(
            nn.Linear(8 * 8 * 512, 4096, bias=True),
            # 8*8*512: output size from encoder (origin img pixel 256*256-> 5 pooling = 8)
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, numCls, bias=True)
        )

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

        self.names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                      'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                      'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
                      'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                      'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5',
                      'linear6', 'relu6', 'drop6', 'linear7', 'relu7', 'drop7', 'linear8']

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits


def load_vgg16(f, device):
    
    input_channels  = 13    # number of bands the VGG16 was trained on 
    classes         = 10    # output units (set to number of classes the VGG16 was trained on)
    net = VGG16(input_channels, classes)

    '''
    if torch.cuda.is_available():
        state_dict = torch.load(f, map_location=device)['model_state_dict']
    else:
        state_dict = torch.load(f, map_location=torch.device('cpu'))['model_state_dict']
    '''
    state_dict = torch.load(f, map_location=device)['model_state_dict']

    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    #net.requires_grad = False

    return net


class LossNetwork(nn.Module):
    """
    Extract certain feature maps from pretrained VGG model, used for computing perceptual loss
    """

    def __init__(self, f, output_layer, device):
        super(LossNetwork, self).__init__()

        self.net = load_vgg16(f, device)
        self.output_layer = output_layer

    def forward(self, x):
        feature_list = []
        for i, (n, module) in enumerate(self.net._modules.items()):
            if n == 'encoder':
                for idx in range(len(module)):
                    x = module[idx](x)
                    if idx in self.output_layer:
                        feature_list.append(x)
            else:
                return feature_list
