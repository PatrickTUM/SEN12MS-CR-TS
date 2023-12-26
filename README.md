# SEN12MS-CR-TS Toolbox 

![banner gif](preview/single_banner.png)
>
> _On average, the majority of all optical satellite data is affected by clouds. This observation shows a scene of agricultural land cover in Czechia from the SEN12MS-CR-TS data set for multi-modal multi-temporal cloud removal. SEN12MS-CR-TS contains whole-year time series of radar and optical satellite data distributed globally across our planet's surface._
----
This repository contains code accompanying the paper
> Ebel, P., Xu, Y., & Schmitt, M. , & Zhu, X. X. (2022). SEN12MS-CR-TS: A Remote Sensing Data Set for Multi-modal Multi-temporal Cloud Removal. IEEE Transactions on Geoscience and Remote Sensing, In Press.

It serves as a quick start for working with the associated SEN12MS-CR-TS data set. For additional information:

* The open-access publication is available at [the IEEE TGRS page](https://ieeexplore.ieee.org/document/9691348). 
* The open-access SEN12MS-CR data set is available at the MediaTUM page [here](https://mediatum.ub.tum.de/1639953) (train split) and [here](https://mediatum.ub.tum.de/1659251) (test split).
* You can find additional information on this and related projects on the associated [cloud removal projects page](https://patrickTUM.github.io/cloud_removal/).
* For any further questions, please reach out to me here or via the credentials on my [website](https://pwjebel.com).
---

## Installation
### Dataset
You can download the SEN12MS-CR-TS data set (or parts of it) via the MediaTUM website [here](https://mediatum.ub.tum.de/1639953) (train split) and [here](https://mediatum.ub.tum.de/1659251) (test split) or in the terminal (passwd: *m1639953* or *m1659251*) using wget or rsync, for instance via

```bash
wget "ftp://m1639953:m1639953@dataserv.ub.tum.de/s1_africa.tar.gz"
rsync -chavzP --stats rsync://m1639953@dataserv.ub.tum.de/m1639953/ .
rsync -chavzP --stats rsync://m1659251@dataserv.ub.tum.de/m1659251/ .
```

For the sake of convenient downloading and unzipping, the data set is sharded into separate archives per sensor modality and geographical region. You can, if needed only download and exclusively work on e.g. Sentinel-2 data for cloud removal in Africa. However, we recommend utilizing the global distribution of ROI and emphasize that this code base is written with the full data set in mind. After all archives are downloaded and their subdirectories extracted (e.g. via `find . -name '*.tar.gz' -exec tar -xzvf {} \;`), you can simply merge them via `rsync -a */* .` in the parent directory to obtain the required structure that the repository's code expects. Handle the test split likewise.

**Update:** You can now easily download SEN12MS-CR-TS (and SEN12MS-CR) via the shell script provided [here](https://github.com/PatrickTUM/SEN12MS-CR-TS/blob/master/util/dl_data.sh).

### Code
Clone this data set via `git clone https://github.com/PatrickTUM/SEN12MS-CR-TS.git`.

The code is written in Python 3 and uses PyTorch > 1.4. It is strongly recommended to run the code with CUDA and GPU support. The code has been developed and deployed in Ubuntu 20 LTS and should be able to run in any comparable OS.

---

## Usage
### Dataset 
If you already have your own model in place or wish to build one on the SEN12MS-CR-TS data loader for training and testing, the data loader can be used as a stand-alone script as demonstrated in `./standalone_dataloader.py`. This only requires the files `./data/dataLoader.py` (the actual data loader) and `./util/detect_cloudshadow.py` (if this type of cloud detector is chosen).

For using the dataset as a stand-alone with your own model, loading multi-temporal multi-modal data from SEN12MS-CR-TS is as simple as

``` python
import torch
from data.dataLoader import SEN12MSCRTS
dir_SEN12MSCRTS = '/path/to/your/SEN12MSCRTS'
sen12mscrts     = SEN12MSCRTS(dir_SEN12MSCRTS, split='all', region='all', n_input_samples=3)
dataloader      = torch.utils.data.DataLoader(sen12mscrts)

for pdx, samples in enumerate(dataloader): print(samples['input'].keys())
```

and, likewise, if you wish to (pre-)train on the mono-temporal multi-modal SEN12MS-CR dataset:
 
``` python
import torch
from data.dataLoader import SEN12MSCR
dir_SEN12MSCR   = '/path/to/your/SEN12MSCR'
sen12mscr       = SEN12MSCR(dir_SEN12MSCR, split='all', region='all')
dataloader      = torch.utils.data.DataLoader(sen12mscr)

for pdx, samples in enumerate(dataloader): print(samples['input'].keys())
```

Depending on your choice of the split, ROI, the length of the input time series and the cloud detector algorithm, you may end up with different samples of input and output data. We encourage making use of as much of the data set as practicable. However, to ensure a well-defined and replicable test split of holdout data on which to benchmark, we provide separate files [here](https://u.pcloud.link/publink/show?code=kZXdbk0ZaAHNV2a5ofbB9UW4xCyCT0YFYAFk) that can be loaded with the `--import_data_path /path/to/files/file.npy` flag. Please use those if you which to report your performances on the test split.

### Basic Commands
You can train a new model via
```bash
python train.py --dataroot /path/to/sen12mscrts --dataset_mode sen12mscrts --name exemplary_training_run --sample_type cloudy_cloudfree --model temporal_branched --netG resnet3d_9blocks_withoutBottleneck --gpu_ids 0 --max_dataset_size 100000 --checkpoints_dir /path/to/results --input_type train --cloud_masks s2cloudless_mask --include_S1 --input_nc 15 --output_nc 13 --G_loss L1 --lambda_GAN 0.0 --display_freq 1000 --alter_initial_model --initial_model_path /path/to/models/baseline_resnet.pth --n_input_samples 3 --region all
```
and you can test a (pre-)trained model via
```bash
python test.py --dataroot /path/to/sen12mscrts --dataset_mode sen12mscrts --results_dir /path/to/results --checkpoints_dir /path/to/results --name exemplary_training_run --model temporal_branched --netG resnet3d_9blocks_withoutBottleneck --include_S1 --input_nc 15 --output_nc 13 --sample_type cloudy_cloudfree --cloud_masks s2cloudless_mask --input_type test --max_dataset_size 100000 --num_test 100000 --n_input_samples 3 --epoch latest --eval --phase test --alter_initial_model --initial_model_path /path/to/models/baseline_resnet.pth --min_cov 0.0 --max_cov 1.0 --region all
```

For a list and description of all flags, please see the parser files in directory `./options`.

---


## References

If you use this code, our models or data set for your research, please cite [this](https://ieeexplore.ieee.org/document/9691348) publication:
```bibtex
@article{sen12mscrts,
        title = {{SEN12MS-CR-TS: A Remote Sensing Data Set for Multi-modal Multi-temporal Cloud Removal}},
        author = {Ebel, Patrick and Xu, Yajin and Schmitt, Michael and Zhu, Xiao Xiang},
        journal = {IEEE Transactions on Geoscience and Remote Sensing},
        year = {2022}
        publisher = {IEEE}
} 
```
You may also be interested in our SEN12MS-CR data set for mono-temporal cloud removal (available [here](https://mediatum.ub.tum.de/1554803)) and the related publication (see [related paper](https://ieeexplore.ieee.org/document/9211498)). Also check out our recently released model for quantifying uncertainties in cloud removal, [UnCRtainTS](https://github.com/PatrickTUM/UnCRtainTS). You can find further information on these and related projects on the accompanying [cloud removal website](https://patrickTUM.github.io/cloud_removal/).



## Credits

This code was originally based on the [STGAN repository](https://github.com/ermongroup/STGAN), which was originally based on the [pix2pix repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Our developed seq2point network was inspired by the original STGAN architecture (see [related paper](https://arxiv.org/abs/1912.06838)) as well as the ResNet for cloud removal in mono-temporal optical satellite data (see [related paper](https://www.sciencedirect.com/science/article/pii/S0924271620301398)). Thanks for making your code publicly available!
