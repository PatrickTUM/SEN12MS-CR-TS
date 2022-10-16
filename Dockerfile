FROM pytorch/pytorch:latest
    
# note: as of now, pytorch/pytorch:latest is not compiled for CUDA > 11.3 yet,
# if you run CUDA > 11.3 please consider base image nvcr.io/nvidia/pytorch:latest
# on NGS: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch

# in case you run CUDA > 11.3 and prefer pytorch/pytorch:latest, then consider this conda-forge build:
# RUN conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

# install dependencies
RUN conda install -c conda-forge cupy  
RUN conda install -c conda-forge opencv
RUN pip install scipy rasterio natsort matplotlib scikit-image tqdm natsort
RUN pip install s2cloudless
RUN conda install pillow=6.1
RUN pip install dominate
RUN pip install visdom

# bake repository into dockerfile
RUN mkdir -p ./data
RUN mkdir -p ./models
RUN mkdir -p ./options
RUN mkdir -p ./util

ADD data ./data
ADD models ./models
ADD options ./options
ADD util ./util
ADD . ./

WORKDIR /workspace
