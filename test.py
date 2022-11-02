"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import warnings
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import torch
from util.visualizer import save_images
from util import html
from util import pytorch_ssim
import models.network_resnet_branched as initial_resnet
import torch.nn as nn
from models.networks_branched import freeze_resnet


def compute_metric(real_B, fake_B, model):
    rmse = torch.sqrt(torch.mean(torch.square(real_B - fake_B)))
    psnr = 20 * torch.log10(1 / rmse)
    mae = torch.mean(torch.abs(real_B - fake_B))
    
    # spectral angle mapper
    mat = real_B * fake_B
    mat = torch.sum(mat, 1)
    mat = torch.div(mat, torch.sqrt(torch.sum(real_B * real_B, 1)))
    mat = torch.div(mat, torch.sqrt(torch.sum(fake_B * fake_B, 1)))
    sam = torch.mean(torch.acos(torch.clamp(mat, -1, 1)))

    ssim = pytorch_ssim.ssim(real_B, fake_B)
    
    # get an aggregated cloud mask over all time points and compute metrics over (non-)cloudy px
    tileTo = real_B.shape[1]
    mask   = torch.clamp(torch.sum(torch.cat(model.A_mask,dim=0), dim=0, keepdim=True), 0, 1)
    mask   = mask.repeat(1, tileTo, 1, 1)
    real_B, fake_B, mask = real_B.cpu().numpy(), fake_B.cpu().numpy(), mask.cpu().numpy()

    rmse_cloudy = np.sqrt(np.nanmean(np.square(real_B[mask==1] - fake_B[mask==1])))
    rmse_cloudfree = np.sqrt(np.nanmean(np.square(real_B[mask==0] - fake_B[mask==0])))
    mae_cloudy = np.nanmean(np.abs(real_B[mask==1] - fake_B[mask==1]))
    mae_cloudfree = np.nanmean(np.abs(real_B[mask==0] - fake_B[mask==0]))

    return {'RMSE': rmse.cpu().numpy().item(),
            'RMSE_cloudy': rmse_cloudy, 
            'RMSE_cloudfree': rmse_cloudfree, 
            'MAE': mae.cpu().numpy().item(),
            'MAE_cloudy': mae_cloudy, 
            'MAE_cloudfree': mae_cloudfree, 
            'PSNR': psnr.cpu().numpy().item(),
            'SAM': sam.cpu().numpy().item(),
            'SSIM': ssim.cpu().numpy().item()}


def save_eval_metric(metric, path, label):
    m = [] # m is a list of lists
    f = open(path+f'/eval_metric_{label}.txt', 'w')

    f.write('Image')
    # for each metric, save the item name
    for index, (name, mat) in enumerate(metric.items()):
        for idx, (crit, value) in enumerate(mat.items()):
            f.write('\t'+crit)
        break
    f.write('\n')
    # for each metric, compute the item value
    for index, (name, mat) in enumerate(metric.items()):
        f.write(name)
        dum = []
        # iterate over each item
        for idx, (crit, value) in enumerate(mat.items()):
            f.write('\t'+str(value))
            dum.append(value)
        f.write('\n')
        m.append(dum)
    f.write('Overall mean')
    # for each metric, compute the average value
    for each in np.nanmean(m, axis=0):
        f.write('\t' + str(each))
    f.close()


def baseline_resnet(opt, device):
    # initiate ResNet model
    m = initial_resnet.ResnetStackedArchitecture(opt)
    # load pre-trained weights
    state_dict = torch.load(opt.initial_model_path)

    # handle state dictionary keys that are eventually misnamed
    if list(state_dict.keys())[0].split('.')[0] != 'model':
        temp_dict = {}
        for key in state_dict.keys():
            temp_dict['.'.join(key.split('.')[1:])] = state_dict[key]
        state_dict = temp_dict

    # assign weights to initiated model
    m.load_state_dict(state_dict)
    # taking the complete model
    model = nn.Sequential(*m.model)
    model.to(device)
    model.eval()
    freeze_resnet(model, False)
    return model

# rescale data to [0, 1], depending on previous range
def scaleTo01(im, method):
    # optionally add clipping of im's range here to minimize potential artifacts
    if method == 'default':
        # rescale from [-1, +1] to [0, 1]
        return (im + 1) / 2
    elif method == 'resnet':
        # dealing with only optical images, range 0,5
        # rescale from [0, 5] to [0, 1]
        return im / 5

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    #opt.num_threads = 10           # test code only supports num_threads = 1
    if opt.batch_size !=1:
        warnings.warn(f'Detected batch size {opt.batch_size}, but only supporting batch size 1! Defaulting to 1')
        opt.batch_size = 1          # test code only supports batch_size = 1 # TODO: change this in future versions
    opt.serial_batches = True       # disable data shuffling
    opt.no_flip = True              # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1             # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)   # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)       # create a model given opt.model and other options
    model.setup(opt)                # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    if opt.eval:
        model.eval() # test with eval mode. This affects layers like batchnorm and dropout.

    # set preprocessing mode
    if opt.benchmark_resnet_model or opt.alter_initial_model:
        preprocessing_method = 'resnet'
        baseline_res = baseline_resnet(opt, model.device)
        print("Benchmarking single time point ResNet or multi time point ResNet-STGAN.") # ... but not STGAN
    else:
        preprocessing_method = 'default'
        print("Benchmarking original STGAN.") # ... but not ResNet

    # set up dictionaries to store performances
    if opt.include_simple_baselines and opt.benchmark_resnet_model:
        baseline_metric = {'base_fake_output': dict(), 'base_mosaic': dict(), 'base_resnet': dict()}
    elif opt.include_simple_baselines:
        baseline_metric = {'base_fake_output': dict(), 'base_mosaic': dict()} 
    elif opt.benchmark_resnet_model:  
        baseline_metric = {'base_resnet': dict()}
    eval_metric = {}

    for i, data in enumerate(dataset):
        # optional early stopping after opt.num_test images.
        if i >= opt.num_test: break
        # skip samples which don't meet the cloud coverage requirements
        if not data["coverage_bin"]:
            print(f"Skipping sample {i}: Did not fit into cloud coverage bin")
            continue

        model.set_input(data)  # unpack data from data loader
        img_path = model.get_image_paths() # get image paths
        if isinstance(img_path[0], tuple): img_path = [img_path[0][0]]
        if i % 10 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))        
        
        # process STGAN / our model
        if not opt.benchmark_resnet_model:
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            im_name = save_images(preprocessing_method, webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, saveTiff=False, savePng=True, image_dir='stgan')
            eval_metric[im_name] = compute_metric(scaleTo01(model.real_B, preprocessing_method), scaleTo01(model.fake_B, preprocessing_method), model)
            if opt.use_perceptual_loss:
                eval_metric[im_name]['PERCEPTUAL LOSS'] = model.get_perceptual_loss(model.netL, model.fake_B, model.real_B).detach().numpy().item()
        else:
            # get a placeholder of the prediction and copy other visuals
            model.fake_B = torch.zeros((opt.output_nc, 256, 256)).to(model.device)
            visuals = model.get_current_visuals()  # get image results

        # compute least cloudy and mosaicing baseline
        if opt.include_simple_baselines:

            # baseline 1: [least cloudy] fake_B = real_A with the least cloud coverage
            least_cloudy_idx = np.argsort([torch.sum(model.A_mask[k]).cpu().numpy() for k in range(opt.n_input_samples)])[0]
            fake_B = model.S2_input[least_cloudy_idx]  # [opt.n_input_samples - 1]
            visuals['fake_B'] = fake_B  # update prediction, export images and evaluate metrics
            im_name = save_images(preprocessing_method, webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, saveTiff=False, savePng=True, image_dir='base_fake_output')
            baseline_metric['base_fake_output'][im_name] = compute_metric(scaleTo01(model.real_B, preprocessing_method), scaleTo01(fake_B, preprocessing_method), model)
            if opt.use_perceptual_loss:
                baseline_metric['base_fake_output'][im_name]['PERCEPTUAL LOSS'] = opt.lambda_percep * model.get_perceptual_loss(model.netL, fake_B, model.real_B).detach().numpy().item()

            # baseline 2: [mosaicing] fake_B = average value of cloudless areas of all the input
            fake_B = torch.tensor(np.nan) * model.fake_B.repeat(opt.n_input_samples, 1, 1, 1)
            for k in range(opt.n_input_samples):
                masked_t  = model.S2_input[k] * (1 - model.A_mask[k])
                masked_t = masked_t.float()
                fake_B[k, masked_t[0] != 0] = masked_t[masked_t != 0]
            # apply mean mosaicing
            if preprocessing_method == "default": # scale to [0, 1] before averaging, afterwards scale back to [-1,1]
                fake_B = torch.tensor(np.nanmean(scaleTo01(fake_B.cpu().numpy(), preprocessing_method), 0, keepdims=True) * 2 - 1).to(model.device)
            elif preprocessing_method == "resnet":
                # scale to [0, 1] before averaging, afterwards scale back to [0, 5]
                fake_B = torch.tensor(np.nanmean(scaleTo01(fake_B.cpu().numpy(), preprocessing_method), 0, keepdims=True) * 5).to(model.device)
            # for pixels that are nan across all time points: take neutral value
            fake_B[torch.isnan(fake_B)] = 0.5
            visuals['fake_B'] = fake_B  # update prediction, export images and evaluate metrics
            im_name = save_images(preprocessing_method, webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, saveTiff=False, savePng=True, image_dir='base_mosaic')
            baseline_metric['base_mosaic'][im_name] = compute_metric(scaleTo01(model.real_B, preprocessing_method), scaleTo01(fake_B, preprocessing_method), model)
            if opt.use_perceptual_loss:
                baseline_metric['base_mosaic'][im_name]['PERCEPTUAL LOSS'] = opt.lambda_percep * model.get_perceptual_loss(model.netL, fake_B, model.real_B).detach().numpy().item()

        # baseline 3: compute resnet baseline
        if opt.benchmark_resnet_model:  # include baseline resnet
            fake_B = baseline_res(torch.cat((model.S2_input[0], data['A_S1'][0].to(model.device)),dim=1))
            visuals['fake_B'] = fake_B  # update prediction, export images and evaluate metrics
            im_name = save_images("resnet", webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, saveTiff=False, savePng=True, image_dir='base_resnet')
            baseline_metric['base_resnet'][im_name] = compute_metric(scaleTo01(model.real_B, "resnet"), scaleTo01(fake_B, "resnet"), model)
            if opt.use_perceptual_loss:
                baseline_metric['base_resnet'][im_name]['PERCEPTUAL LOSS'] = opt.lambda_percep * model.get_perceptual_loss(model.netL, fake_B, model.real_B).detach().numpy().item()

    # export all metrics
    webpage.save()  # save the HTML
    # save metric stats for the STGAN model
    if not opt.benchmark_resnet_model:
        save_eval_metric(eval_metric, web_dir, 'stgan')
        np.save(os.path.join(web_dir, f'eval_metric_{"stgan"}.npy'), eval_metric)
    # save simple baselines and resnet stats
    if opt.include_simple_baselines or opt.benchmark_resnet_model:
        for i, name in enumerate(baseline_metric):
            print(f"Summarizing statistics for baseline {name}")
            save_eval_metric(baseline_metric[name], web_dir, name)
            np.save(os.path.join(web_dir, f'eval_metric_{name}.npy'), baseline_metric[name])
