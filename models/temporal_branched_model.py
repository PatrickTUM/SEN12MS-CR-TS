import torch
from .base_model import BaseModel
from . import networks_branched as networks
import numpy as np
from util import util
import warnings

class TemporalBranchedModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        parser.add_argument('--scramble', type=bool, default=False, help='scramble order of input images?')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight of GAN loss for generator and discriminator')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        '''
        if not opt.include_S1:
            self.visual_names = ['real_A_0', 'A_0_mask', 'real_A_1', 'A_1_mask', 'real_A_2', 'A_2_mask', 'fake_B', 'real_B', 'B_mask']
        else:
            self.visual_names = ['real_A_0', 'A_0_S1', 'A_0_mask', 'real_A_1', 'A_1_S1', 'A_1_mask', 'real_A_2', 'A_2_S1', 'A_2_mask', 'fake_B', 'real_B', 'B_mask']
        '''
        self.opt = opt
        self.visual_names = []
        if opt.include_S1:
            for i in range(opt.n_input_samples):
                self.visual_names.append(f'real_A_{i}')
                self.visual_names.append(f'A_{i}_S1')
                self.visual_names.append(f'A_{i}_mask')
                self.visual_names += ['real_B_S1']
        else:
            for i in range(opt.n_input_samples):
                self.visual_names.append(f'real_A_{i}')
                self.visual_names.append(f'A_{i}_mask')
        self.visual_names = self.visual_names + ['real_B', 'B_mask', 'fake_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain and opt.lambda_GAN != 0:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        #if opt.alter_initial_model:
        #    assert opt.include_S1, Exception('Altering initial model must include S1 data')
        self.netG = networks.define_G(self.device, opt.alter_initial_model, opt.unfreeze_iter, opt.initial_model_path, opt.n_input_samples, opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt=opt)

        if self.isTrain and opt.lambda_GAN != 0:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            # if not 0 -> include GAN loss and decriminator
            self.netD = networks.define_D(opt.n_input_samples*opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            if opt.lambda_GAN != 0:  # not 0 -> include GAN and D
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
                # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

        if opt.use_perceptual_loss:
            # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
            if self.isTrain and opt.lambda_GAN != 0:
                self.loss_names = ['G_GAN', 'G_loss', 'D_real', 'D_fake', 'perceptual']
            else:
                self.loss_names = ['G_loss', 'perceptual']
            assert not opt.vgg16_path == 'none', 'Missing input of VGG16 path.'

            # specify channels to compute perceptual loss over, e.g.
            # [11,20,29] as in DIP code (relu3_1, relu4_1, relu5_1) https://github.com/DmitryUlyanov/deep-image-prior/blob/master/utils/perceptual_loss/perceptual_loss.py
            # ['3', '8', '15'] as in internal video learning ('3': "relu1_2",  '8': "relu2_2", '15': "relu3_3", '22': "relu4_3") https://github.com/Haotianz94/IL_video_inpainting/blob/7bf67772b19f44245495c18f79002fea5853bb57/src/configs/base.py & https://github.com/Haotianz94/IL_video_inpainting/blob/7bf67772b19f44245495c18f79002fea5853bb57/src/models/perceptual.py,
            # ['3', '8', '15', '22'] as in original Gatys et al paper (relu1_2, relu2_2, relu3_3, and relu4_3)
            # [8, 15, 22, 29] is also an option to give a try
            # --> labels correspond to one another

            perceptual_layers = {   'dip': [11, 20, 29],
                                    'video': [3, 8, 15],
                                    'original': [3, 8, 15, 22],
                                    'experimental': [8, 15, 22, 29]
            }

            self.netL = util.LossNetwork(opt.vgg16_path, perceptual_layers[opt.layers_percep], self.device)
        else:
            if self.isTrain and opt.lambda_GAN != 0:
                self.loss_names = ['G_GAN', 'G_loss', 'D_real', 'D_fake']
            else:
                self.loss_names = ['G_loss']
        self.total_iters = 0

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        ''' HARD CODED PART: this processes inputs of hard-coded time points 0, 1, 2
        self.real_A_0 = input['A_0'].to(self.device)
        self.A_0_mask = input['A_0_mask'].numpy()
        self.real_A_1 = input['A_1'].to(self.device)
        self.A_1_mask = input['A_1_mask'].numpy()
        self.real_A_2 = input['A_2'].to(self.device)
        self.A_2_mask = input['A_2_mask'].numpy()
        
                if not self.opt.include_S1:
            self.real_A_input = [self.real_A_0, self.real_A_1, self.real_A_2]
            self.real_A = torch.cat((self.real_A_0, self.real_A_1, self.real_A_2), 1).to(self.device)
            for i in range(3*self.real_A_0.shape[1]):
                self.input_is_SAR.append(False)
        else:
            self.A_0_S1 = input['A_0_S1'].to(self.device)
            self.A_1_S1 = input['A_1_S1'].to(self.device)
            self.A_2_S1 = input['A_2_S1'].to(self.device)
            self.A_0_combined = torch.cat((self.real_A_0, self.A_0_S1), 1).to(self.device)
            self.A_1_combined = torch.cat((self.real_A_1, self.A_1_S1), 1).to(self.device)
            self.A_2_combined = torch.cat((self.real_A_2, self.A_2_S1), 1).to(self.device)
            self.real_A_input = [self.A_0_combined, self.A_1_combined, self.A_2_combined]
            self.real_A = torch.cat((self.A_0_combined, self.A_1_combined, self.A_2_combined), 1).to(self.device)
            for i in range (3):
                for j in range(self.real_A_0.shape[1]):
                    self.input_is_SAR.append(False)
                self.input_is_SAR.append([True, True])
        '''
        
        # dynamically process a variable number of input time points        
        A_input, self.A_mask, self.input_is_SAR = [], [], []
        for i in range(self.opt.n_input_samples):
            A_input.append(input['A_S2'][i].to(self.device))
            setattr(self, f'real_A_{i}', A_input[i])
            self.A_mask.append(input['A_mask'][i].to(self.device))
            setattr(self, f'A_{i}_mask', self.A_mask[i])

        if not self.opt.include_S1:
            self.real_A_input = A_input

            for i in range(self.opt.n_input_samples*A_input[0].shape[1]):
                self.input_is_SAR.append(False)
        else:
            self.real_A_input = []
            for i in range(self.opt.n_input_samples):
                S1 = input['A_S1'][i].to(self.device)
                setattr(self, f'A_{i}_S1', S1)
                self.real_A_input.append(torch.cat((A_input[i], S1), 1).to(self.device))

                for j in range(A_input[i].shape[1]):
                    self.input_is_SAR.append(False)
                self.input_is_SAR.append([True, True])

        # concatenate input patches across time (and across modalities, if including SAR)
        self.real_A = torch.cat(self.real_A_input, 1).to(self.device)
        # bookkeeping of target cloud-free patch
        self.real_B = input['B'].to(self.device)
        # bookkeeping of target mask
        self.B_mask = input['B_mask'].to(self.device)
        if self.opt.include_S1:
            self.real_B_S1 = input['B_S1']

        self.S2_input = A_input

        self.image_paths = input['image_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test> (in base_model.py)."""
        self.fake_B = self.netG(self.real_A_input)  # G(A)
        # self.fake_B = self.netG(self.real_A_0)
        if self.opt.alter_initial_model:
            self.fake_B = 5 * (self.fake_B+1)/2 # need to rescale prediction from [-1,+1] (Tanh() output) to [0,5] (pre-trained ResNet pre-processing)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_GAN
        self.loss_D.backward()

    def criterionG(self, fake_B, real_B):
        if self.opt.G_loss == 'L1':
            loss = torch.nn.L1Loss()
            return loss(fake_B, real_B) * self.opt.lambda_L1
        else:
            raise Exception("Undefined G loss type.")

    def scale_to_01(self, im, method):
        if method == 'default':
            # rescale from [-1, +1] to [0, 1]
            return (im + 1) / 2
        else:
            # dealing with only optical images, range 0,5
            # rescale from [0, 5] to [0, 1]
            return im / 5

    def get_perceptual_loss(self):
        loss = 0.
        if self.opt.alter_initial_model:
            method = 'resnet'
        else:
            method = 'default'
        fake = self.netL(self.scale_to_01(self.fake_B, method))
        real = self.netL(self.scale_to_01(self.real_B, method))
        # pre-trained VGG16 expects input to be in [0, 1],
        # --> ResNet (baseline or initial model) input and target S2 patches are in [0, 5],
        #     STGAN (no pre-trained ResNet) input and target patches are in []
        #     outputs of STGAN (model resnet_9blocks) are in [-1, +1] via Tanh()
        #     outputs of 3D net (model ResnetGenerator3DWithoutBottleneck) are in [-1, +1] via Tanh()

        """
        if self.opt.alter_initial_model:
            # change 
            fake = self.netL(self.fake_B)
            real = self.netL(self.real_B, method)
        else:
            fake = self.netL(self.fake_B)
            real = self.netL(self.real_B, method)
        """
        mse = torch.nn.MSELoss()
        for i in range(len(fake)):
            loss += mse(fake[i], real[i])
        return loss

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        if self.opt.lambda_GAN != 0:
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_loss = self.criterionG(self.fake_B, self.real_B)
        # combine loss and calculate gradients
        if self.opt.use_perceptual_loss:
            self.loss_perceptual = self.get_perceptual_loss() * self.opt.lambda_percep
            if self.opt.lambda_GAN != 0:
                self.loss_G = self.opt.lambda_GAN * self.loss_G_GAN + self.loss_G_loss + self.loss_perceptual
            else:
                self.loss_G = self.loss_G_loss + self.loss_perceptual
        else:
            if self.opt.lambda_GAN != 0:
                self.loss_G = self.opt.lambda_GAN * self.loss_G_GAN + self.loss_G_loss
            else:
                self.loss_G = self.loss_G_loss
        self.loss_G.backward()

    def valid_grad(self, net):
        valid_gradients = True
        for name, param in net.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients: break

        if not valid_gradients:
            warnings.warn(f'detected inf or nan values in gradients. not updating model parameters')
        return valid_gradients

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        if self.opt.lambda_GAN != 0:
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            if self.valid_grad(self.netD):
                self.optimizer_D.step()          # update D's weights
            else:
                self.optimizer_D.zero_grad()  # do not update D
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        # update G
        if self.opt.alter_initial_model:
            if self.opt.gpu_ids == []:
                if self.total_iters >= self.opt.unfreeze_iter and self.netG.initial_freezed:
                    self.set_requires_grad(self.netG.model_initial, True)
                    self.netG.model_initial.train(True)
                    self.netG.initial_freezed = False
            else:
                if self.total_iters >= self.opt.unfreeze_iter and self.netG.module.initial_freezed:
                    self.set_requires_grad(self.netG.module.model_initial, True)
                    self.netG.module.model_initial.train(True)
                    self.netG.module.initial_freezed = False
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        if self.valid_grad(self.netG):
            self.optimizer_G.step()          # update G's weights
        else:
            self.optimizer_G.zero_grad()  # do not update G
