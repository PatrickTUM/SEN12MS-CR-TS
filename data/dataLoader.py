import os
import warnings
import numpy as np
from tqdm import tqdm
from natsort import natsorted

import rasterio
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

# s2cloudless: see https://github.com/sentinel-hub/sentinel2-cloud-detector
from s2cloudless import S2PixelCloudDetector
from util.detect_cloudshadow import get_cloud_mask, get_shadow_mask

""" SEN12MSCRTS data loader class, inherits from torch.utils.data.Dataset

    IN: 
    root:               str, path to your copy of the SEN12MS-CR-TS data set
    split:              str, in [all | train | val | test]
    cloud_masks:        str, type of cloud mask detector to run on optical data, in []
    sample_type:        str, [generic | cloudy_cloudfree]
    n_input_samples:    int, number of input samples in time series
    rescale_method:     str, [default | resnet]
    min_cov:            float, in [0.0, 1.0]
    max_cov:            float, in [0.0, 1.0]
    import_data_path:   str, path to importing the suppl. file specifying what time points to load for input and output
    export_data_path:   str, path to export the suppl. file specifying what time points to load for input and output
    
    OUT:
    data_loader:        SEN12MSCRTS instance, implements an iterator that can be traversed via __getitem__(pdx),
                        which returns the pdx-th dictionary of patch-samples (whose structure depends on sample_type)
"""

class SEN12MSCRTS(Dataset):
    def __init__(self, root, split="all", cloud_masks='s2cloudless_mask', sample_type='cloudy_cloudfree', n_input_samples=3, rescale_method='default', min_cov=0.0, max_cov=1.0, import_data_path=None, export_data_path=None):
        
        self.root_dir = root  # set root directory which contains all ROI
        
        self.ROI            = {'ROIs1158': ['106'],
                               'ROIs1868': ['17', '36', '56', '73', '85', '100', '114', '119', '121', '126', '127', '139', '142', '143'],
                               'ROIs1970': ['20', '21', '35', '40', '57', '65', '71', '82', '83', '91', '112', '116', '119', '128', '132', '133', '135', '139', '142', '144', '149'],
                               'ROIs2017': ['8', '22', '25', '32', '49', '61', '63', '69', '75', '103', '108', '115', '116', '117', '130', '140', '146']}
        
        # define splits conform with SEN12MS-CR
        self.splits         = {}
        all_ROI             = [os.path.join(key, val) for key, vals in self.ROI.items() for val in vals]
        self.splits['test'] = [os.path.join('ROIs1868', '119'), os.path.join('ROIs1970', '139'), os.path.join('ROIs2017', '108'), os.path.join('ROIs2017', '63'), os.path.join('ROIs1158', '106'), os.path.join('ROIs1868', '73'), os.path.join('ROIs2017', '32'),
                               os.path.join('ROIs1868', '100'), os.path.join('ROIs1970', '132'), os.path.join('ROIs2017', '103'), os.path.join('ROIs1868', '142'), os.path.join('ROIs1970', '20'), os.path.join('ROIs2017', '140')]  # official test split, across continents
        self.splits['val']  = [] # insert a validation split here
        self.splits['train']= [roi for roi in all_ROI if roi not in self.splits['val'] and roi not in self.splits['test']]  # all remaining ROI are used for training
        
        self.splits["all"]  = self.splits["train"] + self.splits["test"] + self.splits["val"]
        self.split = split
        
        assert split in ['all', 'train', 'val', 'test'], "Input dataset must be either assigned as all, train, test, or val!"
        assert sample_type in ['generic', 'cloudy_cloudfree'], "Input data must be either generic or cloudy_cloudfree type!"
        assert cloud_masks in [None, 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'], "Unknown cloud mask type!"

        self.modalities     = ["S1", "S2"]
        self.time_points    = range(30)
        self.cloud_masks    = cloud_masks  # e.g. 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'
        self.sample_type    = sample_type if self.cloud_masks is not None else 'generic' # pick 'generic' or 'cloudy_cloudfree'
        self.n_input_t      = n_input_samples  # specifies the number of samples, if only part of the time series is used as an input

        if self.cloud_masks in ['s2cloudless_map', 's2cloudless_mask']:
            self.cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)

        self.import_data_path = import_data_path
        self.export_data_path = export_data_path
        if self.export_data_path: self.data_pairs = {}

        if self.import_data_path:
            # fetch time points as specified in the imported file, expects arguments are set accordingly
            if os.path.isdir(self.import_data_path):
                import_here = os.path.join(self.import_data_path, f'{self.n_input_t}_{self.split}_{self.cloud_masks}.npy')
            else:
                import_here = self.import_data_path
            self.data_pairs = np.load(import_here, allow_pickle=True).item()
            print(f'Importing data pairings for split {self.split} from {import_here}.')

        self.paths          = self.get_paths()
        self.n_samples      = len(self.paths)

        # raise a warning that no data has been found
        if not self.n_samples: self.throw_warn()

        self.method         = rescale_method
        self.min_cov, self.max_cov = min_cov, max_cov

    def throw_warn(self):
        warnings.warn("""No data samples found! Please use the following directory structure:
                        
        path/to/your/SEN12MSCRTS/directory:.
        ├───ROIs1158
        ├───ROIs1868
        ├───ROIs1970
        │   ├───20
        │   ├───21
        │   │   ├───S1
        │   │   └───S2
        │   │       ├───0
        │   │       ├───1
        │   │       │   └─── ... *.tif files
        │   │       └───30
        │   ...
        └───ROIs2017
                        
        Note: the data is provided by ROI geo-spatially separated and sensor modalities individually.
        You can simply merge the downloaded & extracted archives' subdirectories via 'mv */* .' in the parent directory
        to obtain the required structure specified above, which the data loader expects.
        """)

    # indexes all patches contained in the current data split
    def get_paths(self):  # assuming for the same ROI+num, the patch numbers are the same
        print(f'\nProcessing paths for {self.split} split')

        paths = []
        for roi_dir, rois in self.ROI.items():
            for roi in tqdm(rois):
                roi_path = os.path.join(self.root_dir, roi_dir, roi)
                # skip non-existent ROI or ROI not part of the current data split
                if not os.path.isdir(roi_path) or os.path.join(roi_dir, roi) not in self.splits[self.split]: continue
                path_s1_t, path_s2_t = [], [],
                for tdx in self.time_points:
                    # working with directory under time stamp tdx
                    path_s1_complete = os.path.join(roi_path, self.modalities[0], str(tdx))
                    path_s2_complete = os.path.join(roi_path, self.modalities[1], str(tdx))

                    # same as complete paths, truncating root directory's path
                    path_s1 = os.path.join(roi_dir, roi, self.modalities[0], str(tdx))
                    path_s2 = os.path.join(roi_dir, roi, self.modalities[1], str(tdx))

                    # get list of files which contains all the patches at time tdx
                    s1_t = natsorted([os.path.join(path_s1, f) for f in os.listdir(path_s1_complete) if (os.path.isfile(os.path.join(path_s1_complete, f)) and ".tif" in f)])
                    s2_t = natsorted([os.path.join(path_s2, f) for f in os.listdir(path_s2_complete) if (os.path.isfile(os.path.join(path_s2_complete, f)) and ".tif" in f)])

                    # same number of patches
                    assert len(s1_t) == len(s2_t)

                    # sort via file names according to patch number and store
                    path_s1_t.append(s1_t)
                    path_s2_t.append(s2_t)

                # for each patch of the ROI, collect its time points and make this one sample
                for pdx in range(len(path_s1_t[0])):
                    sample = {"S1": [path_s1_t[tdx][pdx] for tdx in self.time_points],
                              "S2": [path_s2_t[tdx][pdx] for tdx in self.time_points]}
                    paths.append(sample)

        return paths

    def read_img(self, path_IMG):
        tif = rasterio.open(path_IMG)
        return tif.read().astype(np.float32)

    def rescale(self, img, oldMin, oldMax):
        oldRange = oldMax - oldMin
        img      = (img - oldMin) / oldRange
        return img

    def process_MS(self, img, method):
        if method=='default':
            intensity_min, intensity_max = 0, 10000                 # define a reasonable range of MS intensities
            img = np.clip(img, intensity_min, intensity_max)        # intensity clipping to a global unified MS intensity range
            img = self.rescale(img, intensity_min, intensity_max)   # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
        if method=='resnet':
            intensity_min, intensity_max = 0, 10000                 # define a reasonable range of MS intensities
            img = np.clip(img, intensity_min, intensity_max)        # intensity clipping to a global unified MS intensity range
            img /= 2000                                             # project to [0,5], preserve global intensities (across patches)
        return img

    def process_SAR(self, img, method):
        if method=='default':
            dB_min, dB_max = -25, 0                                 # define a reasonable range of SAR dB
            img = np.clip(img, dB_min, dB_max)                      # intensity clipping to a global unified SAR dB range
            img = self.rescale(img, dB_min, dB_max)                 # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
        if method=='resnet':
            # project SAR to [0, 2] range
            dB_min, dB_max = [-25.0, -32.5], [0, 0]
            img = np.concatenate([(2 * (np.clip(img[0], dB_min[0], dB_max[0]) - dB_min[0]) / (dB_max[0] - dB_min[0]))[None, ...],
                                  (2 * (np.clip(img[1], dB_min[1], dB_max[1]) - dB_min[1]) / (dB_max[1] - dB_min[1]))[None, ...]], axis=0)
        return img

    def get_cloud_cloudshadow_mask(self, img, cloud_threshold=0.2):
        cloud_mask = get_cloud_mask(img, cloud_threshold, binarize=True)
        shadow_mask = get_shadow_mask(img)

        # encode clouds and shadows as segmentation masks
        cloud_cloudshadow_mask = np.zeros_like(cloud_mask)
        cloud_cloudshadow_mask[shadow_mask < 0] = -1
        cloud_cloudshadow_mask[cloud_mask > 0] = 1

        # label clouds and shadows
        cloud_cloudshadow_mask[cloud_cloudshadow_mask != 0] = 1
        return cloud_cloudshadow_mask

    def get_cloud_mask(self, img):

        if self.cloud_masks == 'cloud_cloudshadow_mask':
            threshold = 0.2  # set to e.g. 0.2 or 0.4
            mask = self.get_cloud_cloudshadow_mask(np.clip(img, 0, 10000), threshold)
        elif self.cloud_masks == 's2cloudless_map':
            threshold = 0.5
            mask = self.cloud_detector.get_cloud_probability_maps(np.moveaxis(np.clip(img, 0, 10000)/10000, 0, -1)[None, ...])[0, ...]
            mask[mask < threshold] = 0
            mask = gaussian_filter(mask, sigma=2).astype(np.float32)
        elif self.cloud_masks == 's2cloudless_mask':
            mask = self.cloud_detector.get_cloud_masks(np.moveaxis(np.clip(img, 0, 10000)/10000, 0, -1)[None, ...])[0, ...]
        else:
            mask = np.ones((img.shape[-1], img.shape[-1]))
        return mask

    def __getitem__(self, pdx):  # get the time series of one patch

        s1          = [self.process_SAR(self.read_img(os.path.join(self.root_dir, img)), self.method) for img in self.paths[pdx]['S1']]
        s2          = [self.read_img(os.path.join(self.root_dir, img)) for img in self.paths[pdx]['S2']]  # note: pre-processing happens after cloud detection
        masks       = None if not self.cloud_masks else [self.get_cloud_mask(img) for img in s2]
        coverage    = [np.mean(mask) for mask in masks]

        # generate data of ((cloudy_t1, cloudy_t2, ..., cloudy_tn), cloud-free) pairings
        # note: filtering the data (e.g. according to cloud coverage etc) and may only use a fraction of the data set
        #       if you wish to train or test on additional samples, then this filtering needs to be adjusted
        if self.sample_type == 'cloudy_cloudfree':
            if self.import_data_path:
                # read indices
                inputs_idx    = self.data_pairs[pdx]['input']
                cloudless_idx = self.data_pairs[pdx]['target']
                target_s1, target_s2, target_mask = np.array(s1)[cloudless_idx], np.array(s2)[cloudless_idx], np.array(masks)[cloudless_idx]
                input_s1, input_s2, input_masks = np.array(s1)[inputs_idx], np.array(s2)[inputs_idx], np.array(masks)[inputs_idx]
                coverage_match = True

            else:  # sample custom time points from the current patch space in the current split
                # sort observation indices according to cloud coverage, ascendingly
                coverage_idx = np.argsort(coverage)
                cloudless_idx = coverage_idx[0]
                # take the (earliest, in case of draw) least cloudy time point as target
                target_s1, target_s2, target_mask = np.array(s1)[cloudless_idx], np.array(s2)[cloudless_idx], np.array(masks)[cloudless_idx]
                # take the first n_input_t samples with cloud coverage e.g. in [0.1, 0.5], ...
                inputs_idx = [pdx for pdx, perc in enumerate(coverage) if perc >= self.min_cov and perc <= self.max_cov][:self.n_input_t]
                coverage_match = True  # assume the requested amount of cloud coverage is met

                if len(inputs_idx) < self.n_input_t:
                    # ... if not exists then take the first n_input_t samples (except target patch)
                    inputs_idx = [pdx for pdx in range(len(coverage)) if pdx!=cloudless_idx][:self.n_input_t]
                    coverage_match = False  # flag input samples that didn't meet the required cloud coverage
                input_s1, input_s2, input_masks = np.array(s1)[inputs_idx], np.array(s2)[inputs_idx], np.array(masks)[inputs_idx]

                if self.export_data_path:
                    # performs repeated writing to file, only use this for processes dedicated for exporting
                    # and if so, only use a single thread of workers (--num_threads 1), this ain't thread-safe
                    self.data_pairs[pdx] = {'input': inputs_idx, 'target': cloudless_idx,
                                            'paths': {'input': {'S1': [self.paths[pdx]['S1'][idx] for idx in inputs_idx],
                                                                'S2': [self.paths[pdx]['S2'][idx] for idx in inputs_idx]},
                                                      'output': {'S1': self.paths[pdx]['S1'][cloudless_idx],
                                                                 'S2': self.paths[pdx]['S2'][cloudless_idx]}}}
                    if os.path.isdir(self.export_data_path):
                        export_here = os.path.join(self.export_data_path, f'{self.n_input_t}_{self.split}_{self.cloud_masks}.npy')
                    else:
                        export_here = export_data_path
                    np.save(export_here, self.data_pairs)

            sample = {'input': {'S1': list(input_s1),
                                'S2': [self.process_MS(img, self.method) for img in input_s2],
                                'masks': list(input_masks),
                                'coverage': [np.mean(mask) for mask in input_masks]
                                },
                      'target': {'S1': [target_s1],
                                 'S2': [self.process_MS(target_s2, self.method)],
                                 'S2_path': os.path.join(self.root_dir, self.paths[pdx]['S2'][cloudless_idx]),
                                 'masks': [target_mask],
                                 'coverage': [np.mean(target_mask)]},
                       'coverage_bin': coverage_match
                      }

        elif self.sample_type == 'generic':  # this returns the whole, unfiltered sequence of S1 & S2 observations
            sample = {'S1': s1,
                      'S2': [self.process_MS(img, self.method) for img in s2],
                      'masks': masks,
                      'coverage': coverage
                      }
        return sample

    def __len__(self):
        # length of generated list
        return self.n_samples
