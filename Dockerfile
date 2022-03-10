FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# install dependencies
RUN conda install -c conda-forge cupy  
RUN conda install -c conda-forge opencv
RUN pip install scipy rasterio natsort matplotlib scikit-image tqdm natsort
RUN pip install s2cloudless
RUN pip install Pillow
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
