# Multimodal Property Valuation
This repository provides code and notebooks to reproduce a property price prediction pipeline using tabular data and satellite imagery.

## Repository Structure

```
PROPERTY_VALUATION/
├── data/
│   ├── images/
│   │   ├── zoom16/
│   │   │   ├── train/
│   │   │   │   └── <property_id>.png
│   │   │   └── test/
│   │   │       └── <property_id>.png
│   │   │
│   │   └── zoom18/
│   │       ├── train/
│   │       │   └── <property_id>.png
│   │       └── test/
│   │           └── <property_id>.png
│   │
│   ├── processed/
│   │   ├── train_processed.csv
│   │   ├── val_processed.csv
│   │   ├── test_processed.csv
│   │   ├── train_with_xgb.csv
│   │   ├── val_with_xgb.csv
│   │   └── test_with_xgb.csv
│   │
│   └── raw/
│       ├── train.xlsx
│       ├── test.xlsx
│       └── images.zip
│
├── outputs/
│   ├── xgboost_model.pkl
│   ├── naive_fusion.pth
│   ├── adaptive_fusion_final.pth
│   ├── learned_model.pth
│   ├── metrics.json
│   └── 24114066_final.csv
│
├── src/
│   ├── dataset.py
│   ├── models.py
│   └── __pycache__/
│
├── visualisation/
│
├── data_fetcher.py
├── preprocess_tabular.ipynb
├── model_training.ipynb
├── predict.ipynb
├── requirements.txt
└── README.md
```

## Image Folder Creation

Before running any files, ensure that the following image directories exist:
```
data/images/zoom16/train/
data/images/zoom16/test/
data/images/zoom18/train/
data/images/zoom18/test/
```
## Environment Setup

Python 3.9 or later is recommended.

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows


pip install -r requirements.txt

```

## Data Sources and Downloads

The tabular property data is provided in the `data/raw/` directory as Excel files (`train.xlsx`, `test.xlsx`) and is converted into processed CSV files during preprocessing.

satellite images need to be downloaded, the image fetching script and API setup instructions are provided below.

## API Setup for Satellite Images

Satellite images are downloaded using a map imagery API (e.g., Google Maps Static API).

To download images from scratch:

1. Create an API key from the provider’s developer console.
   - Google Maps Static API: https://developers.google.com/maps/documentation/maps-static

2. Add the API key in the image fetching script (`data_fetcher.py`) by replacing the placeholder value.

3. Run the image fetching script to download satellite images for each property ID.

Note: API usage may be subject to request limits and billing restrictions depending on the provider.

## Data Preparation

Raw tabular data is provided in the `data/raw/` directory as Excel files.  
To generate the processed CSV files required for training and evaluation, run the preprocessing notebook:   
preprocess_tabular.ipynb


## Path Configuration (Required)

This project uses absolute local paths for data and output directories.  
After cloning the repository, you must update the `BASE_DIR` variable in the following files:

- `model_training.ipynb`
- `predict.ipynb`
- `preprocess_tabular.ipynb`
- `notebooks/eda.ipynb`
- `notebooks/geospatial.ipynb`
- `notebooks/grad_cam.ipynb`
- `notebooks/results_visualisation.ipynb`

In each file, locate the line(it is commented in the file):

```python
BASE_DIR = Path("/Users/your_username/path/to/PROPERTY_VALUATION")


## Running the Code

Run the following notebooks in order to reproduce the complete pipeline:

1. `preprocess_tabular.ipynb` – Prepare processed tabular data  
2. `notebooks/eda.ipynb` – Exploratory data analysis  
3. `notebooks/geospatial.ipynb` – Spatial analysis of property prices  
4. `model_training.ipynb` – Train models and evaluate performance  
5. `notebooks/grad_cam.ipynb` – Generate Grad-CAM visualizations  
6. `notebooks/results_visualisation.ipynb` – Plot performance comparisons  
7. `predict.ipynb` – Generate final predictions on the test set



## Outputs

After running the full pipeline, the following files will be generated in the `outputs/` directory:

- `xgboost_model.pkl` – Trained tabular baseline model  
- `naive_fusion.pth` – Trained naive fusion model  
- `adaptive_fusion_final.pth` – Trained residual fusion CNN  
- `metrics.json` – Validation RMSE and R² scores  
- `24114066_final.csv` – Final predicted prices for the test set  

---

## Notes

- All reported RMSE and R² values correspond to **validation set performance**, not training data.
- Absolute paths (`BASE_DIR`) must be updated before running the code.
- Satellite image downloads may be subject to API usage limits.
- This project is intended for academic and reproducibility purposes.


