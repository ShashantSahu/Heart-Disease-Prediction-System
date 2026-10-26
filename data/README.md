# Dataset folder

CardioScore Pro uses the **UCI Heart Disease dataset, Cleveland subset** (303 patients,
13 input features + 1 outcome).

## What to put here

Save the file as:

```
data/processed.cleveland.data
```

Download page: https://archive.ics.uci.edu/dataset/45/heart+disease
(file name inside the download: `processed.cleveland.data`).

If the folder is empty and you are online, `python train_model.py` tries to download this
file automatically. If that fails, the script prints a clear message; download it manually.

You can also point to any file explicitly:

```
python train_model.py --data path/to/your_file
```

## Accepted format

* Comma-separated, **either** with no header row (the original UCI file) **or** with a header
  row containing these column names:
  `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`
  plus the outcome column named `num` (or `target`, `goal`, `condition`).
* Missing values written as `?` or left empty are fine (they are imputed).
* The outcome `0` means no disease; `1-4` (or `1`) means disease. It is converted to 0/1.

## Required coding (original UCI coding)

| Column | Meaning | Values |
|---|---|---|
| age | Age (years) | number |
| sex | Sex | 0 = female, 1 = male |
| cp | Chest pain type | 1 typical angina, 2 atypical angina, 3 non-anginal, 4 asymptomatic |
| trestbps | Resting blood pressure (mm Hg) | number |
| chol | Serum cholesterol (mg/dl) | number |
| fbs | Fasting blood sugar > 120 mg/dl | 0 = no, 1 = yes |
| restecg | Resting ECG | 0 normal, 1 ST-T abnormality, 2 LV hypertrophy |
| thalach | Maximum heart rate achieved | number |
| exang | Exercise-induced angina | 0 = no, 1 = yes |
| oldpeak | ST depression (exercise vs rest) | number |
| slope | Slope of peak exercise ST segment | 1 up, 2 flat, 3 down |
| ca | Major vessels coloured by fluoroscopy | 0-3 |
| thal | Thallium stress test | 3 normal, 6 fixed defect, 7 reversible defect |

## Important: copies with different coding are rejected on purpose

Many re-published copies (for example some Kaggle versions) use different codes
(`cp` 0-3, `slope` 0-2, flipped `target`). With those, the app's drop-down labels would be
wrong, so `train_model.py` stops with an error naming the column. Use the original UCI file,
or - if your college prescribes a different version - edit the `FEATURES` list at the top of
`train_model.py` so the codes and labels match it. Nothing else needs to change, because the
form, validation, report and tests all read that one list.

## Other files created here

`cardioscore.db` (SQLite) is created automatically on first app start. It stores user
accounts (password hashes only) and assessment history. It is ignored by git. Do not share it.
