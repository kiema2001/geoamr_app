#!/bin/bash
# This script runs automatically on Streamlit Cloud

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# Install NCBI AMRFinderPlus
conda install -c bioconda -c conda-forge ncbi-amrfinderplus -y

# Download latest AMR databases
amrfinder --update

# Verify installation
amrfinder --version
