"# Customer Churn Prediction

A machine learning project for predicting customer churn using a tabular customer dataset. The pipeline loads the dataset, preprocesses features, trains a classifier, and evaluates its performance.

## Project Overview

This repository includes:
- Data preprocessing and feature engineering
- Model training for multiple classifier types
- Model evaluation with classification metrics
- Saved model artifacts in the `models/` directory

## Dataset

The project expects a dataset at:

`data/Churn_Modelling.csv`

The dataset should contain a target column named either `Churn` or `Exited`.

## Supported Models

The training script supports the following models:
- `logistic`
- `random_forest` (default)
- `gradient_boosting`

## Project Structure

```text
customer-churn-prediction/
├── data/
│   └── Churn_Modelling.csv
├── models/
├── notebooks/
├── outputs/
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   └── train.py
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Pipeline

From the project root:

```bash
python main.py
```

Optional arguments:

```bash
python main.py --model_type logistic
python main.py --model_type random_forest
python main.py --model_type gradient_boosting
```

You can also point to a custom dataset:

```bash
python main.py --data_path "path/to/your/data.csv"
```

## Output

The script trains the selected model and saves:
- `models/model.pkl`
- `models/scaler.pkl`

Evaluation metrics are printed to the console, including:
- classification report
- confusion matrix
- ROC-AUC score
- overall accuracy

## Requirements

The project dependencies are listed in `requirements.txt`.

## License

This project is for educational and demonstration purposes.
" 
