"""Dataset class SEN12MSCRTS

This class wraps around the SEN12MSCRTS dataloader in ./dataLoader.py
"""

import numpy as np
import random
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from data.dataLoader import SEN12MSCRTS


class Sen12mscrtsDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self, opt)

        if opt.alter_initial_model or opt.benchmark_resnet_model:
            self.rescale_method = 'resnet' # rescale SAR to [0,2] and optical to [0,5]
        else:
            self.rescale_method = 'default' # rescale all to [-1,1] (gets rescaled to [0,1])

        self.opt 			= opt
        self.data_loader 	= SEN12MSCRTS(opt.dataroot, split=opt.input_type, region=opt.region, cloud_masks=opt.cloud_masks, sample_type=opt.sample_type, n_input_samples=opt.n_input_samples, rescale_method=self.rescale_method, min_cov=opt.min_cov, max_cov=opt.max_cov, import_data_path=opt.import_data_path, export_data_path=opt.export_data_path)
        self.max_bands		= 13

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        
        # call data loader to get item
        cloudy_cloudfree = self.data_loader.__getitem__(index)

        if self.opt.include_S1:
            input_channels = [i for i in range(self.max_bands)]
            
            # for each input sample, collect the SAR data
            A_S1 = []
            for i in range(self.opt.n_input_samples):
                A_S1_01 = cloudy_cloudfree['input']['S1'][i]

                if self.rescale_method == 'default':
                    A_S1.append((A_S1_01 * 2) - 1)  # rescale from [0,1] to [-1,+1]
                elif self.rescale_method == 'resnet':
                    A_S1.append(A_S1_01)  # no need to rescale, keep at [0,2]
                    
            # fetch the target S1 image (and optionally rescale)
            B_S1_01 = cloudy_cloudfree['target']['S1'][0]
            if self.rescale_method == 'default':
                B_S1 = (B_S1_01 * 2) - 1  # rescale from [0,1] to [-1,+1]
            elif self.rescale_method == 'resnet':
                B_S1 = B_S1_01  # no need to rescale, keep at [0,2]
        
        else: # not containing any S1
            assert self.opt.input_nc <= self.max_bands, Exception("MS input channel number larger than 13 (S1 not included)!")
            input_channels = [i for i in range(self.opt.input_nc)]

        # use only NIR+BGR channels when training STGAN
        if self.opt.model == "temporal_branched_ir_modified": input_channels = [7, 1, 2, 3]

        A_S2, A_S2_mask = [], []

        if self.opt.in_only_S1:  # using only S1 input
            input_channels = [i for i in range(self.max_bands)]
            for i in range(self.opt.n_input_samples):
                A_S2_01 = cloudy_cloudfree['input']['S1'][i]
                if self.rescale_method == 'default':
                    A_S2.append((A_S2_01 * 2) - 1)   # rescale from [0,1] to [-1,+1]
                elif self.rescale_method == 'resnet':
                    A_S2.append(A_S2_01)  # no need to rescale, keep at [0,5]
                A_S2_mask.append(cloudy_cloudfree['target']['masks'][0].reshape((1, 256, 256)))
        else: # this is the typical case
            for i in range(self.opt.n_input_samples):
                A_S2_01 = cloudy_cloudfree['input']['S2'][i][input_channels]
                if self.rescale_method == 'default':
                    A_S2.append((A_S2_01 * 2) - 1)  # rescale from 0,1 to -1,+1
                elif self.rescale_method == 'resnet':
                    A_S2.append(A_S2_01)  # no need to rescale, keep at [0,5]
                A_S2_mask.append(cloudy_cloudfree['input']['masks'][i].reshape((1, 256, 256)))

        # get the target cloud-free optical image
        B_01   = cloudy_cloudfree['target']['S2'][0]
        if self.opt.output_nc == 4: B_01 = B_01[input_channels]
        if self.rescale_method == 'default':
            B = (B_01 * 2) - 1  # rescale from [0,1] to [-1,+1]
        elif self.rescale_method == 'resnet':
            B = B_01  # no need to rescale, keep at [0,5]
        B_mask = cloudy_cloudfree['target']['masks'][0].reshape((1, 256, 256))
        image_path = cloudy_cloudfree['target']['S2 path']
        
        coverage_bin = True
        if "coverage bin" in cloudy_cloudfree: coverage_bin = cloudy_cloudfree["coverage bin"]

        if self.opt.include_S1:
            return {'A_S1': A_S1, 'A_S2': A_S2, 'A_mask': A_S2_mask, 'B': B, 'B_S1': B_S1, 'B_mask': B_mask, 'image_path': image_path, "coverage_bin": coverage_bin}
        else:
            return {'A_S2': A_S2, 'A_mask': A_S2_mask, 'B': B, 'B_mask': B_mask, 'image_path': image_path, "coverage_bin": coverage_bin}

    def __len__(self):
        """Return the total number of images."""
        return len(self.data_loader)
