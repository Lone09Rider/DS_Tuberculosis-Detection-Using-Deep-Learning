# Tuberculosis Detection Using Deep Learning

End-to-end scaffold for classifying chest X-ray images into `TB` vs `Normal` using transfer learning (ResNet50, VGG16, EfficientNetB0), with Streamlit UI and AWS-ready deployment hooks. Includes a synthetic metadata CSV to let you exercise the pipeline before pointing to the real dataset.

## Project Outline (from brief)
- Goal: preprocess/augment X-rays, train multiple CNNs, compare metrics (accuracy, precision, recall, F1, ROC-AUC), and deploy a Streamlit front end on AWS.
- Data: 3,008 images (approx.) across `TB` (~2,494) and `Normal` (~514). Provided scaffold uses a small synthetic CSV for structure; swap in the real dataset when available.
- Deliverables: cleaned dataset, trained/evaluated models, Streamlit app for uploads/predictions, deployable on AWS (EC2/Elastic Beanstalk) with saved weights.

## Repository Structure
- `data/` – place datasets here.
  - `raw/tb`, `raw/normal` – drop images following the metadata CSV paths.
  - `synthetic_metadata.csv` – 200-row demo metadata with splits/demographics.
- `models/` – saved weights and training artifacts (created at run time).
- `src/` – core code.
  - `config.py` – central hyperparameters and paths.
  - `data_prep.py` – CSV parsing, tf.data pipelines, augmentations.
  - `training.py` – model builders (ResNet50/VGG16/EfficientNetB0), training loop, checkpoints.
  - `evaluate.py` – metrics/curves and model comparison helpers.
  - `predict.py` – single/multi-image inference utilities.
  - `app.py` – Streamlit UI for uploads and predictions.
- `requirements.txt` – Python dependencies.

## Quickstart
1) Create/activate venv (already configured here) and install deps:
   ```bash
   pip install -r requirements.txt
   ```
2) Point metadata to your real data (optional):
   - Update `data/synthetic_metadata.csv` or drop a new CSV with columns: `patient_id, split, label, image_path, age, sex, view, source`.
   - Ensure `image_path` values exist on disk.
3) Train (example with EfficientNetB0):
   ```bash
   python -m src.training --model efficientnetb0 --epochs 5 --batch-size 16 --csv data/synthetic_metadata.csv --output-dir models/efficientnetb0
   ```
4) Evaluate:
   ```bash
   python -m src.evaluate --csv data/synthetic_metadata.csv --model-path models/efficientnetb0/best.keras --model efficientnetb0
   ```
5) Run Streamlit locally:
   ```bash
   streamlit run src/app.py -- --model-path models/efficientnetb0/best.keras --model efficientnetb0
   ```

## Notes for AWS Deployment
- Package the trained `.keras` file plus `src/` and `requirements.txt` onto EC2/Elastic Beanstalk.
- Expose Streamlit on the instance security group (default port 8501) or front with an ALB.
- For production, place model artifacts in S3 and download on app start (hook in `app.py` placeholder).

## Next Steps
- Swap synthetic CSV with the real dataset.
- Increase epochs, add class-weighting, and enable mixed precision if using GPU.
- Add experiment tracking (Weights & Biases) and monitoring if desired.
