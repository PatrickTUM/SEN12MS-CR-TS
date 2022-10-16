import os
import warnings
import numpy as np
from tqdm import tqdm
from natsort import natsorted

from datetime import datetime
to_date   = lambda string: datetime.strptime(string, '%Y-%m-%d')
S1_LAUNCH = to_date('2014-04-03')

import rasterio
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

# s2cloudless: see https://github.com/sentinel-hub/sentinel2-cloud-detector
from s2cloudless import S2PixelCloudDetector
from util.detect_cloudshadow import get_cloud_mask, get_shadow_mask


# utility functions used in the dataloaders of SEN12MS-CR and SEN12MS-CR-TS
def read_tif(path_IMG):
    tif = rasterio.open(path_IMG)
    return tif

def read_img(tif):
    return tif.read().astype(np.float32)

def rescale(img, oldMin, oldMax):
    oldRange = oldMax - oldMin
    img      = (img - oldMin) / oldRange
    return img

def process_MS(img, method):
    if method=='default':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img = rescale(img, intensity_min, intensity_max)   # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method=='resnet':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img /= 2000                                        # project to [0,5], preserve global intensities (across patches)
    return img

def process_SAR(img, method):
    if method=='default':
        dB_min, dB_max = -25, 0                            # define a reasonable range of SAR dB
        img = np.clip(img, dB_min, dB_max)                 # intensity clipping to a global unified SAR dB range
        img = rescale(img, dB_min, dB_max)                 # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method=='resnet':
        # project SAR to [0, 2] range
        dB_min, dB_max = [-25.0, -32.5], [0, 0]
        img = np.concatenate([(2 * (np.clip(img[0], dB_min[0], dB_max[0]) - dB_min[0]) / (dB_max[0] - dB_min[0]))[None, ...],
                              (2 * (np.clip(img[1], dB_min[1], dB_max[1]) - dB_min[1]) / (dB_max[1] - dB_min[1]))[None, ...]], axis=0)
    return img

def get_cloud_cloudshadow_mask(img, cloud_threshold=0.2):
    cloud_mask = get_cloud_mask(img, cloud_threshold, binarize=True)
    shadow_mask = get_shadow_mask(img)

    # encode clouds and shadows as segmentation masks
    cloud_cloudshadow_mask = np.zeros_like(cloud_mask)
    cloud_cloudshadow_mask[shadow_mask < 0] = -1
    cloud_cloudshadow_mask[cloud_mask > 0] = 1

    # label clouds and shadows
    cloud_cloudshadow_mask[cloud_cloudshadow_mask != 0] = 1
    return cloud_cloudshadow_mask

def get_cloud_map(img, detector, instance=None):

    if detector == 'cloud_cloudshadow_mask':
        threshold = 0.2  # set to e.g. 0.2 or 0.4
        mask = get_cloud_cloudshadow_mask(np.clip(img, 0, 10000), threshold)
    elif detector== 's2cloudless_map':
        threshold = 0.5
        mask = instance.get_cloud_probability_maps(np.moveaxis(np.clip(img, 0, 10000)/10000, 0, -1)[None, ...])[0, ...]
        mask[mask < threshold] = 0
        mask = gaussian_filter(mask, sigma=2).astype(np.float32)
    elif detector == 's2cloudless_mask':
        mask = instance.get_cloud_masks(np.moveaxis(np.clip(img, 0, 10000)/10000, 0, -1)[None, ...])[0, ...]
    else:
        mask = np.ones((img.shape[-1], img.shape[-1]))
    return mask


""" SEN12MSCRTS data loader class, inherits from torch.utils.data.Dataset

    IN: 
    root:               str, path to your copy of the SEN12MS-CR-TS data set
    split:              str, in [all | train | val | test]
    region:             str, [all | africa | america | asiaEast | asiaWest | europa]
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
    def __init__(self, root, split="all", region='all', cloud_masks='s2cloudless_mask', sample_type='cloudy_cloudfree', n_input_samples=3, rescale_method='default', min_cov=0.0, max_cov=1.0, import_data_path=None, export_data_path=None):
        
        self.root_dir = root   # set root directory which contains all ROI
        self.region   = region # region according to which the ROI are selected
        self.ROI      = {'ROIs1158': ['106'],
                         'ROIs1868': ['17', '36', '56', '73', '85', '100', '114', '119', '121', '126', '127', '139', '142', '143'],
                         'ROIs1970': ['20', '21', '35', '40', '57', '65', '71', '82', '83', '91', '112', '116', '119', '128', '132', '133', '135', '139', '142', '144', '149'],
                         'ROIs2017': ['8', '22', '25', '32', '49', '61', '63', '69', '75', '103', '108', '115', '116', '117', '130', '140', '146']}
        
        # define splits conform with SEN12MS-CR
        self.splits         = {}
        if self.region=='all':
            all_ROI             = [os.path.join(key, val) for key, vals in self.ROI.items() for val in vals]
            self.splits['test'] = [os.path.join('ROIs1868', '119'), os.path.join('ROIs1970', '139'), os.path.join('ROIs2017', '108'), os.path.join('ROIs2017', '63'), os.path.join('ROIs1158', '106'), os.path.join('ROIs1868', '73'), os.path.join('ROIs2017', '32'),
                                   os.path.join('ROIs1868', '100'), os.path.join('ROIs1970', '132'), os.path.join('ROIs2017', '103'), os.path.join('ROIs1868', '142'), os.path.join('ROIs1970', '20'), os.path.join('ROIs2017', '140')]  # official test split, across continents
            self.splits['val']  = [os.path.join('ROIs2017', '22'), os.path.join('ROIs1970', '65'), os.path.join('ROIs2017', '117'), os.path.join('ROIs1868', '127'), os.path.join('ROIs1868', '17')] # insert a validation split here
            self.splits['train']= [roi for roi in all_ROI if roi not in self.splits['val'] and roi not in self.splits['test']]  # all remaining ROI are used for training
        elif self.region=='africa':
            self.splits['test'] = [os.path.join('ROIs2017', '32'), os.path.join('ROIs2017', '140')]
            self.splits['val']  = [os.path.join('ROIs2017', '22')]
            self.splits['train']= [os.path.join('ROIs1970', '21'), os.path.join('ROIs1970', '35'), os.path.join('ROIs1970', '40'),
                                   os.path.join('ROIs2017', '8'), os.path.join('ROIs2017', '61'), os.path.join('ROIs2017', '75')]
        elif self.region=='america':
            self.splits['test'] = [os.path.join('ROIs1158', '106'), os.path.join('ROIs1970', '132')]
            self.splits['val']  = [os.path.join('ROIs1970', '65')]
            self.splits['train']= [os.path.join('ROIs1868', '36'), os.path.join('ROIs1868', '85'),
                                   os.path.join('ROIs1970', '82'), os.path.join('ROIs1970', '142'),
                                   os.path.join('ROIs2017', '49'), os.path.join('ROIs2017', '116')]
        elif self.region=='asiaEast':
            self.splits['test'] = [os.path.join('ROIs1868', '73'), os.path.join('ROIs1868', '119'), os.path.join('ROIs1970', '139')]
            self.splits['val']  = [os.path.join('ROIs2017', '117')]
            self.splits['train']= [os.path.join('ROIs1868', '114'), os.path.join('ROIs1868', '126'), os.path.join('ROIs1868', '143'), 
                                   os.path.join('ROIs1970', '116'), os.path.join('ROIs1970', '135'),
                                   os.path.join('ROIs2017', '25')]
        elif self.region=='asiaWest':
            self.splits['test'] = [os.path.join('ROIs1868', '100')]
            self.splits['val']  = [os.path.join('ROIs1868', '127')]
            self.splits['train']= [os.path.join('ROIs1970', '57'), os.path.join('ROIs1970', '83'), os.path.join('ROIs1970', '112'),
                                   os.path.join('ROIs2017', '69'), os.path.join('ROIs1970', '115'), os.path.join('ROIs1970', '130')]
        elif self.region=='europa':
            self.splits['test'] = [os.path.join('ROIs2017', '63'), os.path.join('ROIs2017', '103'), os.path.join('ROIs2017', '108'), os.path.join('ROIs1868', '142'), os.path.join('ROIs1970', '20')]
            self.splits['val']  = [os.path.join('ROIs1868', '17')]
            self.splits['train']= [os.path.join('ROIs1868', '56'), os.path.join('ROIs1868', '121'), os.path.join('ROIs1868', '139'),
                                   os.path.join('ROIs1970', '71'), os.path.join('ROIs1970', '91'), os.path.join('ROIs1970', '119'), os.path.join('ROIs1970', '128'), os.path.join('ROIs1970', '133'), os.path.join('ROIs1970', '144'), os.path.join('ROIs1970', '149'),
                                   os.path.join('ROIs2017', '146')]
        else: raise NotImplementedError

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
        else: self.cloud_detector = None

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
                        
        path/to/your/SEN12MSCRTS/directory:
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
        print(f'\nProcessing paths for {self.split} split of region {self.region}')

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

    def __getitem__(self, pdx):  # get the time series of one patch

        # get images
        s1_tif          = [read_tif(os.path.join(self.root_dir, img)) for img in self.paths[pdx]['S1']]
        s2_tif          = [read_tif(os.path.join(self.root_dir, img)) for img in self.paths[pdx]['S2']]
        coord           = [list(tif.bounds) for tif in s2_tif]
        s1              = [process_SAR(read_img(img), self.method) for img in s1_tif]
        s2              = [read_img(img) for img in s2_tif]  # note: pre-processing happens after cloud detection
        masks           = None if not self.cloud_masks else [get_cloud_map(img, self.cloud_masks, self.cloud_detector) for img in s2]

        # get statistics and additional meta information
        coverage    = [np.mean(mask) for mask in masks]
        s1_dates    = [to_date(img.split('/')[-1].split('_')[5]) for img in self.paths[pdx]['S1']]
        s2_dates    = [to_date(img.split('/')[-1].split('_')[5]) for img in self.paths[pdx]['S2']]
        s1_td       = [(date-S1_LAUNCH).days for date in s1_dates]
        s2_td       = [(date-S1_LAUNCH).days for date in s2_dates]

        # generate data of ((cloudy_t1, cloudy_t2, ..., cloudy_tn), cloud-free) pairings
        # note: filtering the data (e.g. according to cloud coverage etc) and may only use a fraction of the data set
        #       if you wish to train or test on additional samples, then this filtering needs to be adjusted
        if self.sample_type == 'cloudy_cloudfree':
            if self.import_data_path:
                # read indices
                inputs_idx    = self.data_pairs[pdx]['input']
                cloudless_idx = self.data_pairs[pdx]['target']
                target_s1, target_s2, target_mask = np.array(s1)[cloudless_idx], np.array(s2)[cloudless_idx], np.array(masks)[cloudless_idx]
                input_s1, input_s2, input_masks   = np.array(s1)[inputs_idx], np.array(s2)[inputs_idx], np.array(masks)[inputs_idx]
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
                        export_here = self.export_data_path
                    np.save(export_here, self.data_pairs)

            sample = {'input': {'S1': list(input_s1),
                                'S2': [process_MS(img, self.method) for img in input_s2],
                                'masks': list(input_masks),
                                'coverage': [np.mean(mask) for mask in input_masks],
                                'S1 TD': [s1_td[idx] for idx in inputs_idx],
                                'S2 TD': [s2_td[idx] for idx in inputs_idx],
                                'S1 path': [os.path.join(self.root_dir, self.paths[pdx]['S1'][idx]) for idx in inputs_idx],
                                'S2 path': [os.path.join(self.root_dir, self.paths[pdx]['S2'][idx]) for idx in inputs_idx],
                                'coord': [coord[idx] for idx in inputs_idx],
                                },
                      'target': {'S1': [target_s1],
                                 'S2': [process_MS(target_s2, self.method)],
                                 'masks': [target_mask],
                                 'coverage': [np.mean(target_mask)],
                                 'S1 TD': [s1_td[cloudless_idx]],
                                 'S2 TD': [s2_td[cloudless_idx]],
                                 'S1 path': [os.path.join(self.root_dir, self.paths[pdx]['S1'][cloudless_idx])],
                                 'S2 path': [os.path.join(self.root_dir, self.paths[pdx]['S2'][cloudless_idx])],
                                 'coord': [coord[cloudless_idx]],
                                 },
                       'coverage bin': coverage_match
                      }

        elif self.sample_type == 'generic':  # this returns the whole, unfiltered sequence of S1 & S2 observations
            sample = {'S1': s1,
                      'S2': [process_MS(img, self.method) for img in s2],
                      'masks': masks,
                      'coverage': coverage,
                      'S1 TD': s1_td,
                      'S2 TD': s2_td,
                      'S1 path': [os.path.join(self.root_dir, self.paths[pdx]['S1'][idx]) for idx in self.time_points],
                      'S2 path': [os.path.join(self.root_dir, self.paths[pdx]['S2'][idx]) for idx in self.time_points],
                      'coord': coord
                      }
        return sample

    def __len__(self):
        # length of generated list
        return self.n_samples



""" SEN12MSCR data loader class, inherits from torch.utils.data.Dataset

    IN: 
    root:               str, path to your copy of the SEN12MS-CR-TS data set
    split:              str, in [all | train | val | test]
    region:             str, [all | africa | america | asiaEast | asiaWest | europa]
    cloud_masks:        str, type of cloud mask detector to run on optical data, in []
    sample_type:        str, [generic | cloudy_cloudfree]
    n_input_samples:    int, number of input samples in time series
    rescale_method:     str, [default | resnet]
    
    OUT:
    data_loader:        SEN12MSCRTS instance, implements an iterator that can be traversed via __getitem__(pdx),
                        which returns the pdx-th dictionary of patch-samples (whose structure depends on sample_type)
"""

class SEN12MSCR(Dataset):
    def __init__(self, root, split="all", region='all', cloud_masks='s2cloudless_mask', sample_type='pretrain', rescale_method='default'):

        self.root_dir = root   # set root directory which contains all ROI
        self.region   = region # region according to which the ROI are selected # TODO: currently only supporting 'all'
        self.ROI      = {'ROIs1158': ['106'],
                         'ROIs1868': ['17', '36', '56', '73', '85', '100', '114', '119', '121', '126', '127', '139', '142', '143'],
                         'ROIs1970': ['20', '21', '35', '40', '57', '65', '71', '82', '83', '91', '112', '116', '119', '128', '132', '133', '135', '139', '142', '144', '149'],
                         'ROIs2017': ['8', '22', '25', '32', '49', '61', '63', '69', '75', '103', '108', '115', '116', '117', '130', '140', '146']}
        
        # define splits conform with SEN12MS-CR-TS
        self.splits         = {}
        self.splits['train']= ['ROIs1970_fall_s1/s1_3', 'ROIs1970_fall_s1/s1_22', 'ROIs1970_fall_s1/s1_148', 'ROIs1970_fall_s1/s1_107', 'ROIs1970_fall_s1/s1_1', 'ROIs1970_fall_s1/s1_114', 
                               'ROIs1970_fall_s1/s1_135', 'ROIs1970_fall_s1/s1_40', 'ROIs1970_fall_s1/s1_42', 'ROIs1970_fall_s1/s1_31', 'ROIs1970_fall_s1/s1_149', 'ROIs1970_fall_s1/s1_64', 
                               'ROIs1970_fall_s1/s1_28', 'ROIs1970_fall_s1/s1_144', 'ROIs1970_fall_s1/s1_57', 'ROIs1970_fall_s1/s1_35', 'ROIs1970_fall_s1/s1_133', 'ROIs1970_fall_s1/s1_30', 
                               'ROIs1970_fall_s1/s1_134', 'ROIs1970_fall_s1/s1_141', 'ROIs1970_fall_s1/s1_112', 'ROIs1970_fall_s1/s1_116', 'ROIs1970_fall_s1/s1_37', 'ROIs1970_fall_s1/s1_26', 
                               'ROIs1970_fall_s1/s1_77', 'ROIs1970_fall_s1/s1_100', 'ROIs1970_fall_s1/s1_83', 'ROIs1970_fall_s1/s1_71', 'ROIs1970_fall_s1/s1_93', 'ROIs1970_fall_s1/s1_119', 
                               'ROIs1970_fall_s1/s1_104', 'ROIs1970_fall_s1/s1_136', 'ROIs1970_fall_s1/s1_6', 'ROIs1970_fall_s1/s1_41', 'ROIs1970_fall_s1/s1_125', 'ROIs1970_fall_s1/s1_91', 
                               'ROIs1970_fall_s1/s1_131', 'ROIs1970_fall_s1/s1_120', 'ROIs1970_fall_s1/s1_110', 'ROIs1970_fall_s1/s1_19', 'ROIs1970_fall_s1/s1_14', 'ROIs1970_fall_s1/s1_81', 
                               'ROIs1970_fall_s1/s1_39', 'ROIs1970_fall_s1/s1_109', 'ROIs1970_fall_s1/s1_33', 'ROIs1970_fall_s1/s1_88', 'ROIs1970_fall_s1/s1_11', 'ROIs1970_fall_s1/s1_128', 
                               'ROIs1970_fall_s1/s1_142', 'ROIs1970_fall_s1/s1_122', 'ROIs1970_fall_s1/s1_4', 'ROIs1970_fall_s1/s1_27', 'ROIs1970_fall_s1/s1_147', 'ROIs1970_fall_s1/s1_85', 
                               'ROIs1970_fall_s1/s1_82', 'ROIs1970_fall_s1/s1_105', 'ROIs1158_spring_s1/s1_9', 'ROIs1158_spring_s1/s1_1', 'ROIs1158_spring_s1/s1_124', 'ROIs1158_spring_s1/s1_40', 
                               'ROIs1158_spring_s1/s1_101', 'ROIs1158_spring_s1/s1_21', 'ROIs1158_spring_s1/s1_134', 'ROIs1158_spring_s1/s1_145', 'ROIs1158_spring_s1/s1_141', 'ROIs1158_spring_s1/s1_66', 
                               'ROIs1158_spring_s1/s1_8', 'ROIs1158_spring_s1/s1_26', 'ROIs1158_spring_s1/s1_77', 'ROIs1158_spring_s1/s1_113', 'ROIs1158_spring_s1/s1_100', 
                               'ROIs1158_spring_s1/s1_117', 'ROIs1158_spring_s1/s1_119', 'ROIs1158_spring_s1/s1_6', 'ROIs1158_spring_s1/s1_58', 'ROIs1158_spring_s1/s1_120', 'ROIs1158_spring_s1/s1_110', 
                               'ROIs1158_spring_s1/s1_126', 'ROIs1158_spring_s1/s1_115', 'ROIs1158_spring_s1/s1_121', 'ROIs1158_spring_s1/s1_39', 'ROIs1158_spring_s1/s1_109', 'ROIs1158_spring_s1/s1_63', 
                               'ROIs1158_spring_s1/s1_75', 'ROIs1158_spring_s1/s1_132', 'ROIs1158_spring_s1/s1_128', 'ROIs1158_spring_s1/s1_142', 'ROIs1158_spring_s1/s1_15', 'ROIs1158_spring_s1/s1_45', 
                               'ROIs1158_spring_s1/s1_97', 'ROIs1158_spring_s1/s1_147', 'ROIs1868_summer_s1/s1_90', 'ROIs1868_summer_s1/s1_87', 'ROIs1868_summer_s1/s1_25', 'ROIs1868_summer_s1/s1_124', 
                               'ROIs1868_summer_s1/s1_114', 'ROIs1868_summer_s1/s1_135', 'ROIs1868_summer_s1/s1_40', 'ROIs1868_summer_s1/s1_101', 'ROIs1868_summer_s1/s1_42', 
                               'ROIs1868_summer_s1/s1_31', 'ROIs1868_summer_s1/s1_36', 'ROIs1868_summer_s1/s1_139', 'ROIs1868_summer_s1/s1_56', 'ROIs1868_summer_s1/s1_133', 'ROIs1868_summer_s1/s1_55', 
                               'ROIs1868_summer_s1/s1_43', 'ROIs1868_summer_s1/s1_113', 'ROIs1868_summer_s1/s1_100', 'ROIs1868_summer_s1/s1_76', 'ROIs1868_summer_s1/s1_123', 'ROIs1868_summer_s1/s1_143', 
                               'ROIs1868_summer_s1/s1_93', 'ROIs1868_summer_s1/s1_125', 'ROIs1868_summer_s1/s1_89', 'ROIs1868_summer_s1/s1_120', 'ROIs1868_summer_s1/s1_126', 'ROIs1868_summer_s1/s1_72', 
                               'ROIs1868_summer_s1/s1_115', 'ROIs1868_summer_s1/s1_121', 'ROIs1868_summer_s1/s1_146', 'ROIs1868_summer_s1/s1_140', 'ROIs1868_summer_s1/s1_95', 
                               'ROIs1868_summer_s1/s1_102', 'ROIs1868_summer_s1/s1_7', 'ROIs1868_summer_s1/s1_11', 'ROIs1868_summer_s1/s1_132', 'ROIs1868_summer_s1/s1_15', 'ROIs1868_summer_s1/s1_137', 
                               'ROIs1868_summer_s1/s1_4', 'ROIs1868_summer_s1/s1_27', 'ROIs1868_summer_s1/s1_147', 'ROIs1868_summer_s1/s1_86', 'ROIs1868_summer_s1/s1_47', 'ROIs2017_winter_s1/s1_68', 
                               'ROIs2017_winter_s1/s1_25', 'ROIs2017_winter_s1/s1_62', 'ROIs2017_winter_s1/s1_135', 'ROIs2017_winter_s1/s1_42', 'ROIs2017_winter_s1/s1_64', 'ROIs2017_winter_s1/s1_21', 
                               'ROIs2017_winter_s1/s1_55', 'ROIs2017_winter_s1/s1_112', 'ROIs2017_winter_s1/s1_116', 'ROIs2017_winter_s1/s1_8', 'ROIs2017_winter_s1/s1_59', 'ROIs2017_winter_s1/s1_49', 
                               'ROIs2017_winter_s1/s1_104',  'ROIs2017_winter_s1/s1_81', 'ROIs2017_winter_s1/s1_146', 'ROIs2017_winter_s1/s1_75', 
                               'ROIs2017_winter_s1/s1_94', 'ROIs2017_winter_s1/s1_102', 'ROIs2017_winter_s1/s1_61', 'ROIs2017_winter_s1/s1_47']
        self.splits['val']  = ['ROIs2017_winter_s1/s1_22', 'ROIs1868_summer_s1/s1_19', 'ROIs1970_fall_s1/s1_65', 'ROIs1158_spring_s1/s1_17', 'ROIs2017_winter_s1/s1_107', 
                               'ROIs1868_summer_s1/s1_80', 'ROIs1868_summer_s1/s1_127', 'ROIs2017_winter_s1/s1_130', 'ROIs1868_summer_s1/s1_17', 'ROIs2017_winter_s1/s1_84'] 
        self.splits['test'] = ['ROIs1158_spring_s1/s1_106', 'ROIs1158_spring_s1/s1_123', 'ROIs1158_spring_s1/s1_140', 'ROIs1158_spring_s1/s1_31', 'ROIs1158_spring_s1/s1_44', 
                               'ROIs1868_summer_s1/s1_119', 'ROIs1868_summer_s1/s1_73', 'ROIs1970_fall_s1/s1_139', 'ROIs2017_winter_s1/s1_108', 'ROIs2017_winter_s1/s1_63']

        self.splits["all"]  = self.splits["train"] + self.splits["test"] + self.splits["val"]
        self.split = split
        
        assert split in ['all', 'train', 'val', 'test'], "Input dataset must be either assigned as all, train, test, or val!"
        assert sample_type in ['pretrain'], "Input data must be pretrain!"
        assert cloud_masks in [None, 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'], "Unknown cloud mask type!"

        self.modalities     = ["S1", "S2"]
        self.cloud_masks    = cloud_masks   # e.g. 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'
        self.sample_type    = sample_type   # e.g. 'pretrain'

        self.time_points    = range(1)
        self.n_input_t      = 1             # specifies the number of samples, if only part of the time series is used as an input

        if self.cloud_masks in ['s2cloudless_map', 's2cloudless_mask']:
            self.cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)
        else: self.cloud_detector = None

        self.paths          = self.get_paths()
        self.n_samples      = len(self.paths)

        # raise a warning if no data has been found
        if not self.n_samples: self.throw_warn()

        self.method         = rescale_method

    # indexes all patches contained in the current data split
    def get_paths(self):  # assuming for the same ROI+num, the patch numbers are the same
        print(f'\nProcessing paths for {self.split} split of region {self.region}')

        paths = []
        seeds_S1 = natsorted([s1dir for s1dir in os.listdir(self.root_dir) if "_s1" in s1dir])
        for seed in tqdm(seeds_S1):
            rois_S1 = natsorted(os.listdir(os.path.join(self.root_dir, seed)))
            for roi in rois_S1:
                roi_dir  = os.path.join(self.root_dir, seed, roi)
                paths_S1        = natsorted([os.path.join(roi_dir, s1patch) for s1patch in os.listdir(roi_dir)])
                paths_S2        = [patch.replace('/s1', '/s2').replace('_s1', '_s2') for patch in paths_S1]
                paths_S2_cloudy = [patch.replace('/s1', '/s2_cloudy').replace('_s1', '_s2_cloudy') for patch in paths_S1]

                for pdx, _ in enumerate(paths_S1):
                    # omit patches that are potentially unpaired
                    if not all([os.path.isfile(paths_S1[pdx]), os.path.isfile(paths_S2[pdx]), os.path.isfile(paths_S2_cloudy[pdx])]): continue
                    # don't add patch if not belonging to the selected split
                    if not any([split_roi in paths_S1[pdx] for split_roi in self.splits[self.split]]): continue
                    sample = {"S1":         paths_S1[pdx],
                              "S2":         paths_S2[pdx],
                              "S2_cloudy":  paths_S2_cloudy[pdx]}
                    paths.append(sample)
        return paths

    def __getitem__(self, pdx):  # get the triplet of patch with ID pdx
        s1_tif          = read_tif(os.path.join(self.root_dir, self.paths[pdx]['S1']))
        s2_tif          = read_tif(os.path.join(self.root_dir, self.paths[pdx]['S2']))
        s2_cloudy_tif   = read_tif(os.path.join(self.root_dir, self.paths[pdx]['S2_cloudy']))
        coord           = list(s2_tif.bounds)
        s1              = process_SAR(read_img(s1_tif), self.method)
        s2              = read_img(s2_tif)           # note: pre-processing happens after cloud detection
        s2_cloudy       = read_img(s2_cloudy_tif)    # note: pre-processing happens after cloud detection
        mask            = None if not self.cloud_masks else get_cloud_map(s2_cloudy, self.cloud_masks, self.cloud_detector)

        sample = {'input': {'S1': s1,
                            'S2': process_MS(s2_cloudy, self.method),
                            'masks': mask,
                            'coverage': np.mean(mask),
                            'S1 path': os.path.join(self.root_dir, self.paths[pdx]['S1']),
                            'S2 path': os.path.join(self.root_dir, self.paths[pdx]['S2_cloudy']),
                            'coord': coord,
                            },
                    'target': {'S2': process_MS(s2, self.method),
                               'S2 path': os.path.join(self.root_dir, self.paths[pdx]['S2']),
                               'coord': coord,
                                },
                    }
        return sample

    def throw_warn(self):
        warnings.warn("""No data samples found! Please use the following directory structure:

        path/to/your/SEN12MSCR/directory:
            ├───ROIs1158_spring_s1
            |   ├─s1_1
            |   |   |...
            |   |   ├─ROIs1158_spring_s1_1_p407.tif
            |   |   |...
            |    ...
            ├───ROIs1158_spring_s2
            |   ├─s2_1
            |   |   |...
            |   |   ├─ROIs1158_spring_s2_1_p407.tif
            |   |   |...
            |    ...
            ├───ROIs1158_spring_s2_cloudy
            |   ├─s2_cloudy_1
            |   |   |...
            |   |   ├─ROIs1158_spring_s2_cloudy_1_p407.tif
            |   |   |...
            |    ...
            ...

        Note: Please arrange the dataset in a format as e.g. provided by the script dl_data.sh.
        """)

    def __len__(self):
        # length of generated list
        return self.n_samples
