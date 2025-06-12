#!/usr/bin/env bash
set -euo pipefail

# Ensure a clean workspace and necessary directories
mkdir -p data

# download the data
# command below seems to return 404 sometimes, but it should work eventually
kaggle competitions download -c product-recommendation-challenge
mkdir -p tmp
mv product-recommendation-challenge.zip tmp/

# unzip the data
unzip tmp/product-recommendation-challenge.zip -d tmp/
rm -rf tmp/product-recommendation-challenge.zip

# move the data to the data folder
mv tmp/* data/

# remove the tmp folder
rm -rf tmp/