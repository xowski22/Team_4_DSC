# Team 4 @ DSC Recruitment Challenge - Product Recommendation Challenge

## Overview

Team 4's solution for the [Kaggle Product Recommendation Challenge](https://www.kaggle.com/competitions/product-recommendation-challenge/overview).

**Challenge Overview:**
This is a collaborative filtering recommendation system challenge where we predict which products users will interact with based on historical user-item interactions and product metadata.

## Team Members

- [Borys Piwoński](https://github.com/xowski22)
- [Olivier Halupczok](https://github.com/olivierhalupczok)

## Project Structure
- `data/`: Contains the dataset files. Initially empty, populated by `download-data.sh`.
- `docs/`: Contains Jupyter notebooks for analysis and documentation.
  - `data-structure.ipynb`: Detailed exploration of the dataset structure.
  - `eda.ipynb`: Exploratory Data Analysis.
  - `evaluation.ipynb`: Notebook for evaluating models.
- `src/`: Contains the source code.
  - `evaluation.py`: Script for model evaluation.
- `download-data.sh`: Script to download the dataset from Kaggle.
- `readme.md`: This file.

## Dataset Structure
- `train.csv`: User-item interactions with ratings and timestamps
- `test.csv`: Users for whom we need to make predictions  
- `item_metadata.csv`: Product information including categories, prices, descriptions
- `id_mappings.json`: Mappings between original IDs and encoded numeric IDs
- `sample_submission.csv`: Expected submission format (top 10 item recommendations per user)

More information: [docs/data-structure.ipynb](docs/data-structure.ipynb)

## Approach

1. Data exploration and analysis
2. Feature engineering
3. Recommendation model development
4. Model evaluation and submission

## Prerequisites

- Kaggle API
  - `pip install kaggle`
  - more info [in Kaggle API docs](https://www.kaggle.com/docs/api) or [in Kaggle API github repository](https://github.com/Kaggle/kaggle-api)
- Python
- Jupyter Notebook

## How to run project

1. Clone the repository.
2. Install the dependencies using the `requirements.txt` file:
   - `pip install -r requirements.txt`
3. Set up your Kaggle API credentials.
4. Download the data using the `download-data.sh` script:
    - `chmod +x download-data.sh`
    - `./download-data.sh`
5. Explore the notebooks in the `docs/` directory, starting with `eda.ipynb`.