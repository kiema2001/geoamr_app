#!/usr/bin/env bash
set -e

echo "🧬 Initializing Local Linuxbrew Environment..."

# 1. Install Linuxbrew locally inside the runtime instance directory
BREW_DIR="/mount/src/geoamr_app/.linuxbrew"
if [ ! -d "$BREW_DIR" ]; then
    git clone https://github.com/Homebrew/brew "$BREW_DIR/Homebrew"
    mkdir "$BREW_DIR/bin"
    ln -s "../Homebrew/bin/brew" "$BREW_DIR/bin/brew"
fi

# 2. Activate Homebrew paths dynamically
export PATH="$BREW_DIR/bin:$PATH"
eval "$($BREW_DIR/bin/brew shellenv)"

# 3. Add the Bioconda package taps and install AMRFinderPlus
echo "🧪 Installing NCBI AMRFinderPlus via Homebrew..."
brew tap brewsci/bio
brew install amrfinder

# 4. Pull and build the latest CDC/NIH AMR database definitions
echo "🔄 Updating Antimicrobial Resistance Database definitions..."
amrfinder --update

echo "✅ AMRFinderPlus setup is complete!"
