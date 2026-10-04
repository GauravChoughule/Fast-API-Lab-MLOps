# FastAPI Lab – Wine Classifier

Modified version of the FastAPI lab.

**Changes from the original lab**
- Dataset: Iris → **Wine** (13 features, 3 classes)
- Model: Decision Tree → **Random Forest** (200 trees, max_depth 6)
- Stratified train/test split, accuracy + classification report printed on training
- `/predict` now also returns the class name and prediction confidence
- Model is loaded once and cached instead of on every request

## Setup
```bash
python -m venv fastapi_lab_env
source fastapi_lab_env/bin/activate   # Windows: fastapi_lab_env\Scripts\activate
pip install -r requirements.txt
```

## Train
```bash
cd src
python train.py
```

## Run the API
```bash
uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs to test `/predict`.

## Example request
```json
{
  "alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14, "alcalinity_of_ash": 11.2,
  "magnesium": 100, "total_phenols": 2.65, "flavanoids": 2.76,
  "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28,
  "color_intensity": 4.38, "hue": 1.05, "od280_od315": 3.4, "proline": 1050
}
```
Example response: `{"response": 0, "class_name": "class_0", "confidence": 0.98}`
