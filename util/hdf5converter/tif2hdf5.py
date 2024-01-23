from argparse import ArgumentParser
import h5py
import numpy as np
import torch
from tqdm import tqdm
import os, sys
import re

from sen12mscrts_to_hdf5 import SEN12MSCRTS_to_hdf5


def extract_ROI_tile_patch_index(filename):
    ROI, tile, patch = map(re.split('_|.tif', filename.split('/')[-1]).__getitem__, [1, 2, -2])
    return ROI, tile, patch

def create_hdf5_group(hdf5_file, group):   
    with h5py.File(hdf5_file, 'a', libver='latest') as f:   
        if not f.__contains__(group):
            f.create_group(group)

def process_sample_to_hdf5(hdf5_file, batch, verbose=0):  
    if 'S2' in batch:
        ROI, tile, patch = extract_ROI_tile_patch_index(batch['S2_paths'][0][0])

        # Create a hdf5 group: ROI -> tile -> patch -> S2
        create_hdf5_group(hdf5_file, os.path.join(ROI, tile, f'{ROI}_{tile}_patch_{patch}', 'S2'))

        # Populate the group with hdf5 datasets
        with h5py.File(hdf5_file, 'a', libver='latest') as f:    
            
            # S2 image time series, T x C x H x W
            group = os.path.join(ROI, tile, f'{ROI}_{tile}_patch_{patch}', 'S2')
            s2 = torch.cat(batch['S2'], dim=0)
            dset = f[group].create_dataset('S2', data=s2.numpy().astype(np.uint16), compression='gzip', compression_opts=9)

            # Cloud probability mask, T x 1 x H x W
            cloud_prob = torch.cat(batch['cloud_prob'], dim=0).unsqueeze(1)
            dset = f[group].create_dataset('cloud_prob', data=cloud_prob.float(), compression='gzip', compression_opts=9)  
            
            # Cloud mask, T x 1 x H x W
            cloud_mask = torch.cat(batch['cloud_mask'], dim=0).unsqueeze(1)
            dset = f[group].create_dataset('cloud_mask', data=cloud_mask, compression='gzip', compression_opts=9) 
            
            # Date per observation
            dset = f[group].create_dataset('S2_dates', data=[date[0] for date in batch['S2_dates']], compression='gzip', compression_opts=9)      

    if 'S1' in batch:
        ROI, tile, patch = extract_ROI_tile_patch_index(batch['S1_paths'][0][0])

        # Create a hdf5 group: ROI -> tile -> patch -> S1
        create_hdf5_group(hdf5_file, os.path.join(ROI, tile, f'{ROI}_{tile}_patch_{patch}', 'S1'))
        
        # Populate the group with hdf5 datasets
        with h5py.File(hdf5_file, 'a', libver='latest') as f:    
        
            # S1 image time series, T x C x H x W
            group = os.path.join(ROI, tile, f'{ROI}_{tile}_patch_{patch}', 'S1')
            s1 = torch.cat(batch['S1'], dim=0)
            dset = f[group].create_dataset('S1', data=s1, compression='gzip', compression_opts=9)

            # Date per observation
            dset = f[group].create_dataset('S1_dates', data=[date[0] for date in batch['S1_dates']], compression='gzip', compression_opts=9)   

    if verbose == 1:
        print(f'Sample {ROI}_{tile}_patch_{patch} processed.')

        
parser = ArgumentParser()
parser.add_argument('root_source', type=str)
parser.add_argument('split', type=str)
parser.add_argument('region', type=str)
parser.add_argument('root_dest', type=str)
        

def main(args):

    dataset = SEN12MSCRTS_to_hdf5(args.root_source, split=args.split, region=args.region)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    hdf5_file = os.path.join(args.root_dest, args.split + '.hdf5')

    # Create a hdf5 file
    f = h5py.File(hdf5_file, 'a', libver='latest')
    
    # Iterate over all data samples in the given data split: tiff to hdf5 conversion
    for i, batch in enumerate(tqdm(dataloader)):
        process_sample_to_hdf5(hdf5_file, batch, verbose=0)

    f.close()
    print('Done')

    
if __name__ == '__main__':

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    else:
        main(parser.parse_args())
