import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import models.network_resnet_branched as initial_resnet
import math


###############################################################################
# Helper Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(device, alter_initial_model, unfreeze_iter, initial_model_path, n_samples, input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], opt=None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(device, alter_initial_model, n_samples, input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, opt=opt)
    elif netG == 'resnet_9blocks_withoutBottleneck':
        net = ResnetGeneratorWithoutBottleneck(device, alter_initial_model, n_samples, input_nc, output_nc, ngf, use_dropout=use_dropout, n_blocks=9, opt=opt)
    elif netG == 'resnet3d_9blocks_withoutBottleneck':
        net = ResnetGenerator3DWithoutBottleneck(device, alter_initial_model, n_samples, input_nc, output_nc, ngf, use_dropout=use_dropout, n_blocks=9, opt=opt)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(device, alter_initial_model, n_samples, input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, opt=opt)
    elif netG == 'resnet':
        net = Resnet(device, alter_initial_model, n_samples, input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, opt=opt)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetBranchedGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256_independent':
        net = UnetIndependentBranchedGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'independent_resnet_9blocks':
        net = IndependentResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, opt=opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    # initiate the weights of STGAN
    model = init_net(net, init_type, init_gain, gpu_ids)
    # initiate the weights of the pre-trained ResNet, if used
    if netG.find('resnet') != -1 and alter_initial_model:
        if gpu_ids == []:
            m = model.model_initial
        else:
            m = model.module.model_initial
        
        if unfreeze_iter >= 0 and initial_model_path is not None and initial_model_path!='none':  # load from pre-trained model
            state_dict = torch.load(initial_model_path)

            # handle state dictionaries that are misnamed
            if list(state_dict.keys())[0].split('.')[0] != 'model':
                temp_dict = {}
                for key in state_dict.keys():
                    temp_dict['.'.join(key.split('.')[1:])] = state_dict[key]
                state_dict = temp_dict

            m.load_state_dict(state_dict, strict=False)
            print(m.model)
            if netG == 'resnet_9blocks':
                # remove the mapping from 256 to 64 channels, and from 64 to 13 channels
                model_initial = nn.Sequential(*[m.model[i] for i in range(len(m.model) - 2)])
            elif netG == 'resnet':
                model_initial = m #m.model
            else: # e.g. if using 'resnet3d_9blocks_withoutBottleneck'
                model_initial = nn.Sequential(*[m.model[i] for i in range(len(m.model) - (2 + (not opt.no_64C)))]) #(2 + opt.no_64C*2))])#(2 + (not opt.no_64C)))])
            model_initial.eval()
            if unfreeze_iter > 0: freeze_resnet(model_initial)
        else:  # initialise with normal
            if netG == 'resnet_9blocks':
                # remove the mapping from 256 to 64 channels, and from 64 to 13 channels
                model_initial = nn.Sequential(*[m.model[i] for i in range(len(m.model) - 2)])
            elif netG == 'resnet':
                model_initial = m #m.model
            else:
                model_initial = nn.Sequential(*[m.model[i] for i in range(len(m.model) - (2 + (not opt.no_64C)))]) #(2 + opt.no_64C*2))])
        
        if gpu_ids == []:
            model.model_initial = model_initial
        else:
            model.module.model_initial = model_initial

    return model


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class IndependentResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(IndependentResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_initial = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        model_intermediate = []
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model_intermediate += [nn.Conv2d(2 * ngf * mult, 2 * ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(2 * ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model_intermediate += [ResnetBlock(2 * ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model_final = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model_final += [nn.ConvTranspose2d(6 * ngf * mult, int(6 * ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(6 * ngf * mult / 2)),
                      nn.ReLU(True)]
        model_final += [nn.ReflectionPad2d(3)]
        model_final += [nn.Conv2d(6 * ngf, output_nc, kernel_size=7, padding=0)]
        model_final += [nn.Tanh()]

        self.model_initial_0 = nn.Sequential(*model_initial)
        self.model_initial_1 = nn.Sequential(*model_initial)
        self.model_initial_2 = nn.Sequential(*model_initial)
        self.model_intermediate_01 = nn.Sequential(*model_intermediate)
        self.model_intermediate_02 = nn.Sequential(*model_intermediate)
        self.model_intermediate_12 = nn.Sequential(*model_intermediate)
        self.model_final = nn.Sequential(*model_final)

    def forward(self, input):
        """Standard forward"""
        input_0 = input[0]
        input_1 = input[1]
        input_2 = input[2] 
        output_0 = self.model_initial_0(input_0)
        output_1 = self.model_initial_1(input_1)
        output_2 = self.model_initial_2(input_2)
        intermediate_input_01 = torch.cat((output_0, output_1), 1)
        intermediate_input_02 = torch.cat((output_0, output_2), 1)
        intermediate_input_12 = torch.cat((output_1, output_2), 1)
        output_intermediate_0 = self.model_intermediate_01(intermediate_input_01)
        output_intermediate_1 = self.model_intermediate_02(intermediate_input_02)
        output_intermediate_2 = self.model_intermediate_12(intermediate_input_12)
        return self.model_final(torch.cat((output_intermediate_0, output_intermediate_1, output_intermediate_2), 1))	


def factorial(num):
    if num == 0:
        return 1
    elif num > 0:
        f = 1
        for i in range(1, num + 1):
            f *= i
        return f
    else:
        raise Exception("Factorial must be larger than 0")

# freeze the ResNet model's weights
def freeze_resnet(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, device, alter_initial_model, n_samples, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', opt=None):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_initial = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 3
        model_intermediate = []
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model_intermediate += [nn.Conv2d(2 * ngf * mult, 2 * ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(2 * ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model_intermediate += [ResnetBlock(2 * ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model_final = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model_final += [nn.ConvTranspose2d(6 * ngf * mult, int(6 * ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(6 * ngf * mult / 2)),
                      nn.ReLU(True)]
        model_final += [nn.ReflectionPad2d(3)]
        model_final += [nn.Conv2d(6 * ngf, output_nc, kernel_size=7, padding=0)]
        model_final += [nn.Tanh()]

        self.model_initial = nn.Sequential(*model_initial)
        self.model_intermediate = nn.Sequential(*model_intermediate)
        self.model_final = nn.Sequential(*model_final)

    def forward(self, input):
        """Standard forward"""
        input_0 = input[0]
        input_1 = input[1]
        input_2 = input[2]
        output_0 = self.model_initial(input_0)
        output_1 = self.model_initial(input_1)
        output_2 = self.model_initial(input_2)
        intermediate_input_01 = torch.cat((output_0, output_1), 1)
        intermediate_input_02 = torch.cat((output_0, output_2), 1)
        intermediate_input_12 = torch.cat((output_1, output_2), 1)
        output_intermediate_0 = self.model_intermediate(intermediate_input_01)
        output_intermediate_1 = self.model_intermediate(intermediate_input_02)
        output_intermediate_2 = self.model_intermediate(intermediate_input_12)
        return self.model_final(torch.cat((output_intermediate_0, output_intermediate_1, output_intermediate_2), 1))

class Resnet(nn.Module):
    """baseline ResNet
    """

    def __init__(self, device, alter_initial_model, n_samples, input_nc, output_nc, ngf=64, use_dropout=False, n_blocks=6, padding_type='reflect', norm_layer=None, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(Resnet, self).__init__()

        assert alter_initial_model, Exception("Without bottleneck only works with initial ResNet model. Please use flags --alter_initial_model and --initial_model_path, after fetching the model via: wget https://syncandshare.lrz.de/dl/fiFfN2bj6DaFXfGEGAaAvdZE/baseline_resnet.pth")
        assert n_samples == 1

        # get initial model, see network_resnet_branched.py
        # 16 blocks of 256 features, then mapping from 256 to 64 channels, and finally from 64 to 13 channels
        # (note: the last two CONV layers may or may not be taken according to line:
        # model_initial = nn.Sequential(*[m.model[i] for i in range(len(m.model) - 2)]))
        model_initial = initial_resnet.ResnetStackedArchitecture(opt=opt)
        model_initial.to(device)
        self.initial_freezed = True
        self.n_channels = 256 if not opt else opt.resnet_F
        self.model_initial = model_initial

    def forward(self, input):
        """Standard forward"""

        """
        initial_output = []
        for each in input:
            initial_output.append(self.model_initial(each))
        x = torch.stack(initial_output, dim=2)
        output = self.model_final(x)
        """
        output = self.model_initial(input[0])
        return output

class ResnetGeneratorWithoutBottleneck(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, device, alter_initial_model, n_samples, input_nc, output_nc, ngf=64, use_dropout=False, n_blocks=6, padding_type='reflect', opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGeneratorWithoutBottleneck, self).__init__()

        assert alter_initial_model,  Exception("Without bottleneck only works with initial ResNet model. Please use flags --alter_initial_model and --initial_model_path, after fetching the model via: wget https://syncandshare.lrz.de/dl/fiFfN2bj6DaFXfGEGAaAvdZE/baseline_resnet.pth")
        assert n_samples >= 3

        model_initial = initial_resnet.ResnetStackedArchitecture(opt=opt)
        model_initial.to(device)
        self.initial_freezed = True
        self.n_channels = 256 if not opt else opt.resnet_F
        
        model_intermediate, model_final = [], []
        resnet_blocks = 4
        for i in range(resnet_blocks):
            model_intermediate += [ResnetBlock(2 * self.n_channels, padding_type=padding_type, norm_layer=False, use_dropout=False, use_bias=True, res_scale=1.0)]

        n_channel = 2 * int(factorial(n_samples) / (factorial(n_samples - 2) * factorial(2)))
        for i in range(resnet_blocks):
            model_final += [ResnetBlock(n_channel * self.n_channels, padding_type=padding_type, norm_layer=False, use_dropout=False, use_bias=True, res_scale=1.0)]

        model_final += [nn.Conv2d(n_channel * self.n_channels, output_nc, kernel_size=3, padding=1)]
        model_final += [nn.Tanh()]

        self.model_initial = model_initial
        self.model_intermediate = nn.Sequential(*model_intermediate)
        self.model_final = nn.Sequential(*model_final)
        self.n_samples = n_samples

    def forward(self, input):
        """Standard forward"""
        output, intermediate_input, output_intermediate = [], [], []
        for each in input:
            output.append(self.model_initial(each))
        for i in range(self.n_samples - 1):
            for j in range(i + 1, self.n_samples):
                intermediate_input.append(torch.cat((output[i], output[j]), 1))
        for each in intermediate_input:
            output_intermediate.append(self.model_intermediate(each))
        return self.model_final(torch.cat(output_intermediate, 1))

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, res_scale = 1.0):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.res_scale = res_scale
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if norm_layer == False:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), nn.ReLU(True)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if norm_layer == False:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), nn.ReLU(True)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.res_scale * self.conv_block(x)  # add skip connections
        return out

class ResnetGenerator3DWithoutBottleneck(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, device, alter_initial_model, n_samples, input_nc, output_nc, ngf=64, use_dropout=False, n_blocks=6, padding_type='reflect', opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator3DWithoutBottleneck, self).__init__()

        assert alter_initial_model,  Exception("Without bottleneck only works with initial ResNet model. Please use flags --alter_initial_model and --initial_model_path, after fetching the model via: wget https://syncandshare.lrz.de/dl/fiFfN2bj6DaFXfGEGAaAvdZE/baseline_resnet.pth")
        assert n_samples >= 3

        model_initial = initial_resnet.ResnetStackedArchitecture(opt=opt)
        model_initial.to(device)
        self.initial_freezed = True
        self.n_channels = 256 if not opt else opt.resnet_F

        model_final = [nn.Conv3d(self.n_channels, self.n_channels, kernel_size=(3, 3, 3), padding=1, bias=True), nn.ReLU(True)]
        for i in range(5):
            model_final += [ResnetBlock3D(self.n_channels, padding_type=padding_type, norm_layer='none', use_bias=True, res_scale=0.1)]

        model_final += [ReflectionPad3D(0, 1)]
        model_final += [nn.Conv3d(self.n_channels, output_nc, kernel_size=(n_samples, 3, 3), padding=0)]

        model_final += [nn.Tanh()]

        self.model_initial = model_initial
        self.model_final = nn.Sequential(*model_final)
        self.n_samples = n_samples

    def forward(self, input):
        """Standard forward"""
        initial_output = []
        for each in input:
            initial_output.append(self.model_initial(each))
        x = torch.stack(initial_output, dim=2)
        output = self.model_final(x)
        return output.squeeze(0).transpose(0, 1)

class ReflectionPad3D(nn.Module):
    def __init__(self, pad_D, pad_HW):
        super(ReflectionPad3D, self).__init__()
        self.padder_HW = nn.ReflectionPad2d(pad_HW)
        self.padder_D = nn.ReplicationPad3d((0, 0, 0, 0, pad_D, pad_D))

    def forward(self, x):
        assert (x.size(0) == 1)  # 1 x C x D x H x W
        y = x.squeeze(0).transpose(0, 1)  # D x C x H x W
        y = self.padder_HW(y)
        y = y.transpose(0, 1).unsqueeze(0)  # 1 x C x D x H x W
        return self.padder_D(y)

class BatchNorm3D(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm3D, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        assert (x.size(0) == 1)  # 1 x C x D x H x W
        y = x.squeeze(0).transpose(0, 1).contiguous()  # D x C x H x W
        y = self.bn(y)
        y = y.transpose(0, 1).unsqueeze(0)
        return y

class ResnetBlock3D(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_bias, res_scale = 1.0, late_relu=True):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock3D, self).__init__()
        self.res_scale = res_scale
        self.late_relu = late_relu
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, possibly normalisation layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [ReflectionPad3D(1, 1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if norm_layer == 'BatchNorm3D':
            conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), BatchNorm3D(dim), nn.ReLU(True)]
        else:
            conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), nn.ReLU(True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [ReflectionPad3D(1, 1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if norm_layer == 'BatchNorm3D':
            conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), BatchNorm3D(dim)]
        else:
            conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]

        if self.late_relu:
            self.block_output_relu = nn.ReLU(True)
        else:
            conv_block += [nn.ReLU(True)]


        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.res_scale * self.conv_block(x)  # add skip connections
        if self.late_relu: out = self.block_output_relu(out)
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()

        '''
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        '''

        self.down8 = UnetSkipConnectionBlockDown(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        self.down7 = UnetSkipConnectionBlockDown(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.down6 = UnetSkipConnectionBlockDown(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.down5 = UnetSkipConnectionBlockDown(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.down4 = UnetSkipConnectionBlockDown(ngf * 4, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.down3 = UnetSkipConnectionBlockDown(ngf * 2, ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.down2 = UnetSkipConnectionBlockDown(ngf, ngf * 2, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.down1 = UnetSkipConnectionBlockDown(output_nc, ngf, input_nc=input_nc, submodule=None, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        
        self.up8 = UnetSkipConnectionBlockUp(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        self.up7 = UnetSkipConnectionBlockUp(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.up6 = UnetSkipConnectionBlockUp(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.up5 = UnetSkipConnectionBlockUp(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.up4 = UnetSkipConnectionBlockUp(ngf * 4, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.up3 = UnetSkipConnectionBlockUp(ngf * 2, ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.up2 = UnetSkipConnectionBlockUp(ngf, ngf * 2, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.up1 = UnetSkipConnectionBlockUp(output_nc, ngf, input_nc=input_nc, submodule=None, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        

    def forward(self, x):
        """Standard forward"""
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)
        x = self.up8(x8, x7)
        x = self.up7(x, x6)
        x = self.up6(x, x5)
        x = self.up5(x, x4)
        x = self.up4(x, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x)
        return x

class UnetSkipConnectionBlockDown(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlockDown, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down
            else:
                model = down

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class UnetSkipConnectionBlockUp(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlockUp, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = up + [nn.Dropout(0.5)]
            else:
                model = up

        self.model = nn.Sequential(*model)

    def forward(self, x1, x2):
        if self.outermost:
            return self.model(x1)
        else:   # add skip connections
            return torch.cat([x2, self.model(x1)], 1)

class UnetIndependentBranchedGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetIndependentBranchedGenerator, self).__init__()

        '''
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        '''

        self.down8 = UnetIndependentBranchedSkipConnectionBlockDown(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        self.down7 = UnetIndependentBranchedSkipConnectionBlockDown(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.down6 = UnetIndependentBranchedSkipConnectionBlockDown(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.down5 = UnetIndependentBranchedSkipConnectionBlockDown(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.down4 = UnetIndependentBranchedSkipConnectionBlockDown(ngf * 4, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.down3 = UnetIndependentBranchedSkipConnectionBlockDown(ngf * 2, ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.down2 = UnetIndependentBranchedSkipConnectionBlockDown(ngf, ngf * 2, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.down1 = UnetIndependentBranchedSkipConnectionBlockDown(output_nc, ngf, input_nc=input_nc, submodule=None, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        
        self.up8 = UnetBranchedSkipConnectionBlockUp(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        self.up7 = UnetBranchedSkipConnectionBlockUp(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.up6 = UnetBranchedSkipConnectionBlockUp(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.up5 = UnetBranchedSkipConnectionBlockUp(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.up4 = UnetBranchedSkipConnectionBlockUp(ngf * 4, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.up3 = UnetBranchedSkipConnectionBlockUp(ngf * 2, ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.up2 = UnetBranchedSkipConnectionBlockUp(ngf, ngf * 2, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.up1 = UnetBranchedSkipConnectionBlockUp(output_nc, ngf, input_nc=input_nc, submodule=None, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        

    def forward(self, x):
        """Standard forward"""
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)
        x = self.up8(x8, x7)
        x = self.up7(x, x6)
        x = self.up6(x, x5)
        x = self.up5(x, x4)
        x = self.up4(x, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x)
        return x

class UnetIndependentBranchedSkipConnectionBlockDown(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetIndependentBranchedSkipConnectionBlockDown, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down
            else:
                model = down

        self.model_0 = nn.Sequential(*model)
        self.model_1 = nn.Sequential(*model)
        self.model_2 = nn.Sequential(*model)

    def forward(self, x):
        if self.innermost:
            return torch.cat([self.model_0(x[0]), self.model_1(x[1]), self.model_2(x[2])], 1)
        return [self.model_0(x[0]), self.model_1(x[1]), self.model_2(x[2])]

class UnetBranchedGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetBranchedGenerator, self).__init__()

        '''
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        '''

        self.down8 = UnetBranchedSkipConnectionBlockDown(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        self.down7 = UnetBranchedSkipConnectionBlockDown(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.down6 = UnetBranchedSkipConnectionBlockDown(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.down5 = UnetBranchedSkipConnectionBlockDown(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.down4 = UnetBranchedSkipConnectionBlockDown(ngf * 4, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.down3 = UnetBranchedSkipConnectionBlockDown(ngf * 2, ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.down2 = UnetBranchedSkipConnectionBlockDown(ngf, ngf * 2, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.down1 = UnetBranchedSkipConnectionBlockDown(output_nc, ngf, input_nc=input_nc, submodule=None, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        
        self.up8 = UnetBranchedSkipConnectionBlockUp(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        self.up7 = UnetBranchedSkipConnectionBlockUp(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.up6 = UnetBranchedSkipConnectionBlockUp(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.up5 = UnetBranchedSkipConnectionBlockUp(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)  # add the innermost layer
        self.up4 = UnetBranchedSkipConnectionBlockUp(ngf * 4, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.up3 = UnetBranchedSkipConnectionBlockUp(ngf * 2, ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.up2 = UnetBranchedSkipConnectionBlockUp(ngf, ngf * 2, input_nc=None, submodule=None, norm_layer=norm_layer)
        self.up1 = UnetBranchedSkipConnectionBlockUp(output_nc, ngf, input_nc=input_nc, submodule=None, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        

    def forward(self, x):
        """Standard forward"""
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)
        x = self.up8(x8, x7)
        x = self.up7(x, x6)
        x = self.up6(x, x5)
        x = self.up5(x, x4)
        x = self.up4(x, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x)
        return x

class UnetBranchedSkipConnectionBlockDown(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetBranchedSkipConnectionBlockDown, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down
            else:
                model = down

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.innermost:
            return torch.cat([self.model(x[0]), self.model(x[1]), self.model(x[2])], 1)
        return [self.model(x[0]), self.model(x[1]), self.model(x[2])]

class UnetBranchedSkipConnectionBlockUp(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetBranchedSkipConnectionBlockUp, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(2*inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = up
        elif innermost:
            upconv = nn.ConvTranspose2d(3*inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = up
        else:
            upconv = nn.ConvTranspose2d(2*inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = up + [nn.Dropout(0.5)]
            else:
                model = up

        self.model = nn.Sequential(*model)

    def forward(self, x1, x2):
        if self.outermost:
            return self.model(x1)
        else:   # add skip connections
            return torch.cat([x2[0], x2[1], x2[2], self.model(x1)], 1)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
