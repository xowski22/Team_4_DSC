# Team 4 @ DSC Recruitment Challenge - Product Recommendation Challenge

## Overview

Team 4's solution for the [Kaggle Product Recommendation Challenge](https://www.kaggle.com/competitions/product-recommendation-challenge/overview).

**Challenge Overview:**
This is a collaborative filtering recommendation system challenge where we predict which products users will interact with based on historical user-item interactions and product metadata.

## Team Members

- [Borys Piwoński](https://github.com/xowski22)
- [Olivier Halupczok](https://github.com/olivierhalupczok)

## Files
- `data/`: Dataset files
- `eda.ipynb`: Exploratory Data Analysis
- `readme.md`: Project documentation
- `download-data.sh`: Script to download the data from Kaggle

## Dataset Structure
- `train.csv`: User-item interactions with ratings and timestamps
- `test.csv`: Users for whom we need to make predictions  
- `item_metadata.csv`: Product information including categories, prices, descriptions
- `id_mappings.json`: Mappings between original IDs and encoded numeric IDs
- `sample_submission.csv`: Expected submission format (top 10 item recommendations per user)

## Approach

1. Data exploration and analysis
2. Feature engineering
3. Recommendation model development
4. Model evaluation and submission

## Prerequisites

- Kaggle API
  - `pip install kaggle`
  - more info [here](https://www.kaggle.com/docs/api) or [here](https://github.com/Kaggle/kaggle-api)
- Python 3.10
- Jupyter Notebook

## How to run project

1. Clone the repository
2. Install the dependencies
3. Download the data using the `download-data.sh` script:
    - `chmod +x download-data.sh`
    - `./download-data.sh`
4. run the notebooks