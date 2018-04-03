# use Caffe2 image as parent image
FROM caffe2/caffe2:snapshot-py2-cuda9.0-cudnn7-ubuntu16.04

# apt-get 
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg wget vim 

# python dependencies (conda)
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh

ENV CONDA=/opt/conda/bin/conda

RUN $CONDA update --yes conda && \
    $CONDA install -c menpo -y opencv3 && \
    $CONDA install -y numpy cython scipy matplotlib pyyaml jupyter protobuf future mock setuptools

ENV PATH=/opt/conda/bin:$PATH

# python dependencies (pip)
RUN pip install sk-video

# caffe2
RUN mv /usr/local/caffe2 /usr/local/caffe2_build

ENV Caffe2_DIR /usr/local/caffe2_build
ENV PYTHONPATH /usr/local/caffe2_build:${PYTHONPATH}
ENV LD_LIBRARY_PATH /usr/local/caffe2_build/lib:${LD_LIBRARY_PATH}

# COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

# clone and build detectron
RUN git clone https://github.com/suhangpro/detectron /detectron
WORKDIR /detectron/lib
RUN sed -i -e 's/python2/python/g' Makefile && \
    make && \
    make ops

# entrypoint
WORKDIR /detectron
ENTRYPOINT ["python", "tools/infer_simple_img.py"]
CMD ["demo"]
