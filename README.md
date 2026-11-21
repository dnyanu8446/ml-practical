# Wine Classification MLOps

End-to-end multiclass classification on the Scikit-learn Wine dataset. DVC
versions the source dataset and generated pipeline artifacts; a local offline
feature store manages the feature table and schema; MLflow tracks and compares
three classifiers.

## 1. Set up the environment

Run these commands in PowerShell from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The source `wine.csv` is already tracked by DVC. If you need to regenerate it,
run `python load_data.py` from this folder and then track the new data version:

```powershell
python -m dvc add wine.csv
git add wine.csv.dvc .gitignore
git commit -m "Update wine dataset version"
```

## 2. Run the DVC pipeline

```powershell
python -m dvc repro
python -m dvc metrics show
```

The `build_features` stage writes `data/processed/wine_features.csv` and
`feature_store/schema.json`. The offline feature store assigns each input row a
stable `wine_id`, records the feature/target schema, validates the table, and
keeps target labels separate from the model feature list. Both artifacts are
outputs of the DVC stage, so DVC can reproduce and version them.

The `train` stage compares logistic regression, random forest, and SVC using
stratified cross-validation on the training split. It selects the model with
the highest mean CV accuracy, evaluates it on a held-out test split, saves the
model to `models/best_model.joblib`, and records test/CV metrics in
`metrics/summary.json`. Split and model settings are in `params.yaml`.

Each candidate model is also logged as a separate run in the MLflow
`wine-classification` experiment. View the runs with:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000> in a browser. The local SQLite database and
artifacts under `mlruns/` are ignored by Git; set `MLFLOW_TRACKING_URI` if you
want to use a different tracking server.

## 3. Run predictions

Provide a CSV with the 13 wine feature columns (the `target` column is optional):

```powershell
python -m src.predict --input path\to\wine_features.csv --output predictions.csv
```

## 4. Version and push data with DVC

The `.dvc` file versions the dataset pointer in Git; the actual data is stored
in DVC's cache. Configure a DVC remote once (replace the example with your own
S3, Azure, SSH, or other supported storage URL):

```powershell
python -m dvc remote add -d storage "<REMOTE_URL>"
git add .dvc\config
git commit -m "Configure DVC remote"
```

Push cached data and pipeline outputs to that remote with:

```powershell
python -m dvc push
```

Do not commit cloud credentials. Use the provider's normal credential
configuration or DVC's supported credential mechanism.

## 5. Push the project code with Git

This folder is a standalone Git repository. Review the changes, stage the
pipeline files, commit, and push the branch to your GitHub repository:

```powershell
git status
git add .gitignore README.md dvc.yaml dvc.lock params.yaml requirements.txt wine.csv.dvc
git add data/processed/.gitignore feature_store/.gitignore metrics/.gitignore models/.gitignore
git add src/__init__.py src/feature_store.py src/predict.py src/train.py tests/test_feature_store.py
git commit -m "Build wine classification MLOps pipeline"
git push -u origin HEAD
```

If `origin` is not configured in your clone, add your GitHub repository URL
before pushing:

```powershell
git remote add origin https://github.com/<OWNER>/<REPOSITORY>.git
```

The GitHub repository stores code and DVC metadata. The DVC remote stores the
dataset and generated artifacts; `git push` does not replace `dvc push`.

## Reproduce from a fresh clone

After cloning and installing dependencies, configure the same DVC remote, then
run:

```powershell
python -m dvc pull
python -m dvc repro
```
