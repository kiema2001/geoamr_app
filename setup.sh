#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🧬 Starting GeoAMR Custom Bio-Environment Setup..."

# 1. Create a local bin directory inside your app directory
mkdir -p /mount/src/geoamr_app/bin
cd /mount/src/geoamr_app/bin

# 2. Download the latest official NCBI AMRFinderPlus pre-compiled binary for Linux
echo "📥 Fetching latest AMRFinderPlus binaries from NCBI..."
wget -q https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/AMRFinderPlus/binaries/latest/amrfinder_latest_amd64.tar.gz

# 3. Extract the binaries
echo "📦 Extracting binaries..."
tar -xzf amrfinder_latest_amd64.tar.gz
rm amrfinder_latest_amd64.tar.gz

# 4. Add the local bin folder to the system PATH temporarily so the app can find it
export PATH="/mount/src/geoamr_app/bin:$PATH"

# 5. Download and compile the latest CDC/NIH AMR database
echo "🔄 Initializing and updating NCBI AMR Database..."
./amrfinder --update

echo "✅ GeoAMR Environment Setup Complete!"
