# (Boring) House Price Prediciton 
Predict housing price using regression models and build an interactive dashboard with Streamlit. 

## Project Overview
This project uses the **Ames Housing Dateset** from Kaggle Competition to build a machine learnig model that predicts home prices based on various features like square footage, number of rooms, and location.
The final result is deployed in a dashboard using **Streamlit**.

## Goals
- Explore data and visualize key patterns
- Clean and preprocess the dataset
- Train multiple models and evaluate performance
- Tune and select the best model
- Create an interactive dashboard for prediction

## Dataset
- Source: [Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Rows: 1460
- Features: 79+
- Target: `SalePrice`

## Tech Stack
- Language: Python
- Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn
- Dashboard: Streamlit
- Version Control: Git, GitHub
- Deployment: Streamlit Cloud (tentative)


## Key Results
### Exploratory Data Analysis (EDA)


### Model Comparison


### Final Model




## Project Structure
- `notebooks/`: EDA & modeling notebooks
- `models/`: Several useless models, final trained model (`joblib`) and feature list (`json`)
- `data/`: Raw and cleaned data files
- `outputs/`: Generated charts and figures
- `app/`: Streamlit app
- `requirements.txt`: Python dependencies

## How to Run This Project
```bash
# Clone this repo
git clone https://github.com/irdazh/house-price-prediction.git

# Create a virtual env and install requirements
python -m venv .venv
source .venv/bin/activate #or .venv\Scripts\activate on Windows
pip install -r requirements.txt 

# Run notebooks in order
jupyter notebook

# Run the streamlit app
streamlit run app/app.py
```
## Sample Dashboard

## Author
Daud M. Azhari   
[GitHub](https://github.com/irdazh) | 
[LinkedIn](https:///www.linkedin.com/in/daud-ma)


## Status
- Done: 
- In Progress: 
- Final Dashboard + Deployment by **Week 2**


### Another Tutorial (Author's note: ignore it)
```bash
# as a starter
mkdir house-price-prediction
cd house-price-prediction

# init the git, then
mkdir app data models notebooks outputs scripts

# create venv
python -m venv .venv
source .venv/Scripts/activate #or .venv/bin/activate
pip install [list of packages]
```