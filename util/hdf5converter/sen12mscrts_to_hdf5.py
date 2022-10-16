# scripts kindly provided by Corinne Stucker
# https://scholar.google.ch/citations?user=P-op4CgAAAAJ&hl=de
# this code can be used to reconstruct the full-scene images in hdf5 format from the released individual patches in tif format

from natsort import natsorted
import numpy as np
import os
import rasterio
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from s2cloudless import S2PixelCloudDetector

from data.dataLoader import SEN12MSCRTS

""" SEN12MSCRTS data loader class, used to load the data in the original format and prepare the data for hdf5 export

    IN: 
    root:               str, path to your copy of the SEN12MS-CR-TS data set
    split:              str, in [all | train | val | test]
    region:             str, [all | africa | america | asiaEast | asiaWest | europa]
    cloud_masks:        str, type of cloud mask detector to run on optical data, in [None | cloud_cloudshadow_mask | s2cloudless_map | s2cloudless_mask]

    OUT:
    data_loader:        SEN12MSCRTS instance, implements an iterator that can be traversed via __getitem__(pdx),
                        which returns the pdx-th dictionary of patch-samples (whose structure depends on sample_type)
"""


class SEN12MSCRTS_to_hdf5(SEN12MSCRTS):
    def __init__(self, root, split="all", region='all', cloud_masks='s2cloudless_mask', modalities=["S1", "S2"]):

        self.root_dir = root  # set root directory which contains all ROI
        self.region = region  # region according to which the ROI are selected
        self.ROI = {'ROIs1158': ['106'],
                    'ROIs1868': ['17', '36', '56', '73', '85', '100', '114', '119', '121', '126', '127', '139', '142',
                                 '143'],
                    'ROIs1970': ['20', '21', '35', '40', '57', '65', '71', '82', '83', '91', '112', '116', '119', '128',
                                 '132', '133', '135', '139', '142', '144', '149'],
                    'ROIs2017': ['8', '22', '25', '32', '49', '61', '63', '69', '75', '103', '108', '115', '116', '117',
                                 '130', '140', '146']}

        # define splits conform with SEN12MS-CR
        self.splits = {}
        if self.region == 'all':
            all_ROI = [os.path.join(key, val) for key, vals in self.ROI.items() for val in vals]
            self.splits['test'] = [os.path.join('ROIs1868', '119'), os.path.join('ROIs1970', '139'),
                                   os.path.join('ROIs2017', '108'), os.path.join('ROIs2017', '63'),
                                   os.path.join('ROIs1158', '106'), os.path.join('ROIs1868', '73'),
                                   os.path.join('ROIs2017', '32'),
                                   os.path.join('ROIs1868', '100'), os.path.join('ROIs1970', '132'),
                                   os.path.join('ROIs2017', '103'), os.path.join('ROIs1868', '142'),
                                   os.path.join('ROIs1970', '20'),
                                   os.path.join('ROIs2017', '140')]  # official test split, across continents
            self.splits['val'] = [os.path.join('ROIs2017', '22'), os.path.join('ROIs1970', '65'),
                                  os.path.join('ROIs2017', '117'), os.path.join('ROIs1868', '127'),
                                  os.path.join('ROIs1868', '17')]  # insert your favorite validation split here
            self.splits['train'] = [roi for roi in all_ROI if roi not in self.splits['val'] and roi not in self.splits[
                'test']]  # all remaining ROI are used for training
        elif self.region == 'africa':
            self.splits['test'] = [os.path.join('ROIs2017', '32'), os.path.join('ROIs2017', '140')]
            self.splits['val'] = [os.path.join('ROIs2017', '22')]
            self.splits['train'] = [os.path.join('ROIs1970', '21'), os.path.join('ROIs1970', '35'),
                                    os.path.join('ROIs1970', '40'),
                                    os.path.join('ROIs2017', '8'), os.path.join('ROIs2017', '61'),
                                    os.path.join('ROIs2017', '75')]
        elif self.region == 'america':
            self.splits['test'] = [os.path.join('ROIs1158', '106'), os.path.join('ROIs1970', '132')]
            self.splits['val'] = [os.path.join('ROIs1970', '65')]
            self.splits['train'] = [os.path.join('ROIs1868', '36'), os.path.join('ROIs1868', '85'),
                                    os.path.join('ROIs1970', '82'), os.path.join('ROIs1970', '142'),
                                    os.path.join('ROIs2017', '49'), os.path.join('ROIs2017', '116')]
        elif self.region == 'asiaEast':
            self.splits['test'] = [os.path.join('ROIs1868', '73'), os.path.join('ROIs1868', '119'),
                                   os.path.join('ROIs1970', '139')]
            self.splits['val'] = [os.path.join('ROIs2017', '117')]
            self.splits['train'] = [os.path.join('ROIs1868', '114'), os.path.join('ROIs1868', '126'),
                                    os.path.join('ROIs1868', '143'),
                                    os.path.join('ROIs1970', '116'), os.path.join('ROIs1970', '135'),
                                    os.path.join('ROIs2017', '25')]
        elif self.region == 'asiaWest':
            self.splits['test'] = [os.path.join('ROIs1868', '100')]
            self.splits['val'] = [os.path.join('ROIs1868', '127')]
            self.splits['train'] = [os.path.join('ROIs1970', '57'), os.path.join('ROIs1970', '83'),
                                    os.path.join('ROIs1970', '112'),
                                    os.path.join('ROIs2017', '69'), os.path.join('ROIs1970', '115'),
                                    os.path.join('ROIs1970', '130')]
        elif self.region == 'europa':
            self.splits['test'] = [os.path.join('ROIs2017', '63'), os.path.join('ROIs2017', '103'),
                                   os.path.join('ROIs2017', '108'),
                                   os.path.join('ROIs1868', '142'), os.path.join('ROIs1970', '20')]
            self.splits['val'] = [os.path.join('ROIs1868', '17')]
            self.splits['train'] = [os.path.join('ROIs1868', '56'), os.path.join('ROIs1868', '121'),
                                    os.path.join('ROIs1868', '139'),
                                    os.path.join('ROIs1970', '71'), os.path.join('ROIs1970', '91'),
                                    os.path.join('ROIs1970', '119'),
                                    os.path.join('ROIs1970', '128'), os.path.join('ROIs1970', '133'),
                                    os.path.join('ROIs1970', '144'),
                                    os.path.join('ROIs1970', '149'),
                                    os.path.join('ROIs2017', '146')]
        else:
            raise NotImplementedError

        self.splits["all"] = self.splits["train"] + self.splits["test"] + self.splits["val"]
        self.split = split

        assert split in ['all', 'train', 'val',
                         'test'], "Input dataset must be either assigned as all, train, test, or val!"
        assert cloud_masks in [None, 'cloud_cloudshadow_mask', 's2cloudless_map',
                               's2cloudless_mask'], "Unknown cloud mask type!"

        self.modalities = modalities
        self.time_points = range(30)
        self.cloud_masks = cloud_masks  # e.g. 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'

        if self.cloud_masks in ['s2cloudless_map', 's2cloudless_mask']:
            self.cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)

        self.paths = self.get_paths()
        self.n_samples = len(self.paths)

        # raise a warning that no data has been found
        if not self.n_samples: self.throw_warn()

    def get_paths(self):  # assuming for the same ROI+num, the patch numbers are the same
        print(f'\nProcessing paths for {self.split} split of region {self.region}')

        paths = []
        for roi_dir, rois in self.ROI.items():
            for roi in tqdm(rois):
                roi_path = os.path.join(self.root_dir, roi_dir, roi)
                # skip non-existent ROI or ROI not part of the current data split
                if not os.path.isdir(roi_path) or os.path.join(roi_dir, roi) not in self.splits[self.split]: continue
                path_s1_t, path_s2_t = [], []
                for tdx in self.time_points:
                    if 'S1' in self.modalities:
                        path_s1_complete = os.path.join(roi_path, 'S1', str(tdx))
                        path_s1 = os.path.join(roi_dir, roi, 'S1', str(tdx))
                        s1_t = natsorted([os.path.join(path_s1, f) for f in os.listdir(path_s1_complete) if
                                          (os.path.isfile(os.path.join(path_s1_complete, f)) and ".tif" in f)])
                    if 'S2' in self.modalities:
                        path_s2_complete = os.path.join(roi_path, 'S2', str(tdx))
                        path_s2 = os.path.join(roi_dir, roi, 'S2', str(tdx))
                        s2_t = natsorted([os.path.join(path_s2, f) for f in os.listdir(path_s2_complete) if
                                          (os.path.isfile(os.path.join(path_s2_complete, f)) and ".tif" in f)])

                    if 'S1' in self.modalities and 'S2' in self.modalities:
                        # same number of patches
                        assert len(s1_t) == len(s2_t)

                    # sort via file names according to patch number and store
                    if 'S1' in self.modalities:
                        path_s1_t.append(s1_t)
                    if 'S2' in self.modalities:
                        path_s2_t.append(s2_t)

                # for each patch of the ROI, collect its time points and make this one sample
                for pdx in range(len(path_s1_t[0])):
                    sample = dict()
                    if 'S1' in self.modalities:
                        sample['S1'] = [path_s1_t[tdx][pdx] for tdx in self.time_points]
                    if 'S2' in self.modalities:
                        sample['S2'] = [path_s2_t[tdx][pdx] for tdx in self.time_points]

                    paths.append(sample)

        return paths

    def get_cloud_mask(self, img, mask_type):
        if mask_type == 'cloud_cloudshadow_mask':
            threshold = 0.2  # set to e.g. 0.2 or 0.4
            mask = self.get_cloud_cloudshadow_mask(np.clip(img, 0, 10000), threshold)
        elif mask_type == 's2cloudless_map':
            threshold = 0.5
            mask = self.cloud_detector.get_cloud_probability_maps(np.moveaxis(np.clip(img, 0, 10000)/10000, 0, -1)[None, ...])[0, ...]
            mask[mask < threshold] = 0
            mask = gaussian_filter(mask, sigma=2).astype(np.float32)
        elif mask_type == 's2cloudless_mask':
            mask = self.cloud_detector.get_cloud_masks(np.moveaxis(np.clip(img, 0, 10000)/10000, 0, -1)[None, ...])[0, ...]
        elif mask_type == 's2cloud_prob':
            mask = self.cloud_detector.get_cloud_probability_maps(np.moveaxis(np.clip(img, 0, 10000) / 10000, 0, -1)[None, ...])[0, ...]

        return mask

    def __getitem__(self, pdx):  # get the time series of one patch

        sample = dict()

        if 'S1' in self.modalities:
            s1 = [self.read_img(os.path.join(self.root_dir, img)) for img in self.paths[pdx]['S1']]
            s1_dates = [img.split('/')[-1].split('_')[5] for img in self.paths[pdx]['S1']]
            sample['S1'] = s1
            sample['S1_dates'] = s1_dates
            sample['S1_paths'] = self.paths[pdx]['S1']

        if 'S2' in self.modalities:
            s2 = [self.read_img(os.path.join(self.root_dir, img)) for img in self.paths[pdx]['S2']]
            s2_dates = [img.split('/')[-1].split('_')[5] for img in self.paths[pdx]['S2']]
            
            cloud_prob = [self.get_cloud_mask(img, 's2cloud_prob') for img in s2]
            cloud_mask = [self.get_cloud_mask(img, 's2cloudless_mask') for img in s2]

            sample['S2'] = s2
            sample['S2_dates'] = s2_dates
            sample['S2_paths'] = self.paths[pdx]['S2']
            sample['cloud_prob'] = cloud_prob
            
            sample['cloud_mask'] = cloud_mask

        return sample

    def __len__(self):
        # length of generated list
        return self.n_samples
