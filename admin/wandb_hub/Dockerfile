FROM ubuntu:18.04

# Install basics
RUN apt-get update
RUN apt-get install -y build-essential  # For being able to compile bottleneck and some other Python package
RUN apt-get install -y curl             # For downloading conda
RUN apt-get install -y git              # We need git!
RUN apt-get install -y shellcheck       # For linting

# Install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install git-lfs && git-lfs install

# Install miniconda
RUN cd /tmp && curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -p ~/miniconda -b && rm Miniconda3-latest-Linux-x86_64.sh && cd -
ENV PATH=$HOME/miniconda/bin:$PATH

# Install base conda packages to speed things up
RUN conda update -y conda
RUN conda install cudatoolkit=10.1 cudnn=7.6 pip python=3.7

# Install some heavy packages in the default conda environment simply to cache these packages and save time later
RUN pip install tensorflow==2.2.0rc2 torch
