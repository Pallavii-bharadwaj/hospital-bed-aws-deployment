# Hospital Bed Occupancy Predictor — Deployed on AWS EC2

Predicts 30-day hospital bed occupancy from patient clinical features using Ridge Regression. Trained on real coursework data (MSc Data Science, University of York), deployed as a live REST API on AWS EC2 (London region).

## Live API
- Health check: http://16.61.238.74:8000/health
- API docs: http://16.61.238.74:8000/docs

## Model
- Dataset: 2,400 patient records with clinical features (age, CCI, CRP, creatinine, etc.)
- Pipeline: ColumnTransformer (StandardScaler + OneHotEncoder) + Ridge Regression
- Selection: GridSearchCV with repeated 5-fold cross-validation across OLS, Ridge, Lasso, ElasticNet
- Results: Test RMSE = 1.50 bed-days, R² = 0.678

## Stack
- Python, FastAPI, Scikit-learn, Uvicorn
- Deployed on AWS EC2 t3.micro (eu-west-2, London)
- REST API with /health and /predict endpoints

## Run locally
```bash
pip install -r requirements.txt
python3 train_model.py
uvicorn main:app --host 0.0.0.0 --port 8000
```
