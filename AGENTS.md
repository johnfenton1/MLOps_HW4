# HW4 MLOps Run Notes

## Detected Dataset

The local dataset detected for this project is:

`../olist_cleaned_integrated_dataset (3).csv`

You can override it with:

```powershell
$env:DATA_PATH="C:\path\to\olist_cleaned_integrated_dataset.csv"
```

## Finish Criteria

- `model/model.pkl` exists and loads from `app.py`.
- `model/model_metadata.json` records the exact feature list, categorical values, metrics, and detected dataset.
- `train_and_serialize.py` runs successfully.
- Flask serves `/health`, `/predict`, and `/predict/batch`.
- `test_api.py` passes all 5 required tests.
- Docker image builds and the container passes the same API tests.
- Both notebooks are complete for submission review.

## Local Commands

```powershell
cd "C:\Users\jfent\Downloads\New folder\MLOps\hw4-mlops"
python -m pip install -r requirements.txt
python train_and_serialize.py
python app.py
```

In a second terminal:

```powershell
cd "C:\Users\jfent\Downloads\New folder\MLOps\hw4-mlops"
python test_api.py
```

## MLflow Commands

```powershell
cd "C:\Users\jfent\Downloads\New folder\MLOps\hw4-mlops"
mlflow ui --backend-store-uri .\mlruns
```

Open `http://127.0.0.1:5000` for screenshots. If Flask is already using port 5000, run:

```powershell
mlflow ui --backend-store-uri .\mlruns --port 5001
```

## Docker Commands

```powershell
cd "C:\Users\jfent\Downloads\New folder\MLOps\hw4-mlops"
docker build -t hw4-api .
docker run --rm -p 5000:5000 hw4-api
```

In a second terminal:

```powershell
cd "C:\Users\jfent\Downloads\New folder\MLOps\hw4-mlops"
python test_api.py
```

## Deployed API Test

```powershell
$env:API_BASE_URL="https://your-render-service.onrender.com"
python test_api.py
```
