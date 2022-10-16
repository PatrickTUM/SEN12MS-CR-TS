#!/usr/bin/env bash

# scripts kindly provided by Corinne Stucker
# https://scholar.google.ch/citations?user=P-op4CgAAAAJ&hl=de
# this code can be used to reconstruct the full-scene images in hdf5 format from the released individual patches in tif format

#python tif2hdf5.py /scratch2/Data/SEN12MS-CR-TS/ val europa /scratch2/Data/SEN12MS-CR-TS_hdf5/all_ROIs/
#python tif2hdf5.py /scratch2/Data/SEN12MS-CR-TS/ val america /scratch2/Data/SEN12MS-CR-TS_hdf5/all_ROIs/
#python tif2hdf5.py /scratch2/Data/SEN12MS-CR-TS/ val africa /scratch2/Data/SEN12MS-CR-TS_hdf5/all_ROIs/

python tif2hdf5.py /scratch2/Data/SEN12MS-CR-TS/ val asiaWest /scratch2/Data/SEN12MS-CR-TS_hdf5/all_ROIs/
python tif2hdf5.py /scratch2/Data/SEN12MS-CR-TS/ val asiaEast /scratch2/Data/SEN12MS-CR-TS_hdf5/all_ROIs/
#python tif2hdf5.py /scratch2/Data/SEN12MS-CR-TS_testSplit/ test europa /scratch2/Data/SEN12MS-CR-TS_hdf5/all_ROIs/
#python tif2hdf5.py /scratch2/Data/SEN12MS-CR-TS_testSplit/ test america /scratch2/Data/SEN12MS-CR-TS_hdf5/all_ROIs/
#python tif2hdf5.py /scratch2/Data/SEN12MS-CR-TS_testSplit/ test africa /scratch2/Data/SEN12MS-CR-TS_hdf5/all_ROIs/
#python tif2hdf5.py /scratch2/Data/SEN12MS-CR-TS_testSplit/ test asiaWest /scratch2/Data/SEN12MS-CR-TS_hdf5/all_ROIs/
#python tif2hdf5.py /scratch2/Data/SEN12MS-CR-TS_testSplit/ test asiaEast /scratch2/Data/SEN12MS-CR-TS_hdf5/all_ROIs/
