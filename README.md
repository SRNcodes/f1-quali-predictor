# F1 Qualifying Predictor

Predicts F1 qualifying pace using historical + current-season data from the
Jolpica-F1 API, normalized as "pace delta to pole" so times are comparable
across circuits and years.

## Pipeline stages

1. **ingestion/** — pulls raw qualifying data from the Jolpica-F1 API, saves
   to `data/raw/`.
2. **features/** — normalizes lap times into pace deltas, builds a leak-free
   training set (only uses prior-race form to predict each race), saves to
   `data/processed/`.
3. **training/** — compares Ridge / Random Forest / XGBoost / Gaussian
   Process via cross-validation, saves the best model to `models/`.
4. **inference/** — (in progress) FastAPI app serving predictions over HTTP.

## Running locally (no Docker yet)

```bash
cd ingestion && pip install -r requirements.txt
python fetch_data.py --last-year 2025 --this-year 2026 --circuit baku

cd ../features && pip install -r requirements.txt
python build_features.py --last-year 2025 --this-year 2026 --circuit baku

cd ../training && pip install -r requirements.txt
python train.py
```

## Roadmap

- [x] Data ingestion from Jolpica-F1
- [x] Feature engineering (pace delta normalization)
- [x] Model comparison + training
- [ ] FastAPI inference endpoint
- [ ] Dockerize each stage
- [ ] Local Kubernetes (minikube/kind)
- [ ] Deploy on AWS EKS + S3 + ECR
- [ ] CI/CD via GitHub Actions
