# Setup Instructions

## Create virtual environment

python -m venv venv

## Activate environment

source venv/bin/activate

## Install dependencies

pip install -r requirements.txt

## Download dataset

python src/data_download.py

## Preprocess data

python src/data_preprocessing.py

## Train model

python src/train.py

## Track experiments

python src/experiment.py

## Package pipeline

python src/package_model.py
