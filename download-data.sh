#!/usr/bin/env bash
set -euo pipefail

# Create data directory
mkdir -p data

# Download competition data
kaggle competitions download -c product-recommendation-challenge

# Extract directly to data directory
unzip -o product-recommendation-challenge.zip -d data/

# Clean up
rm product-recommendation-challenge.zip