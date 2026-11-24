# DineScope

## Restaurant Intelligence & Analytics

**Understand Restaurants. Discover Patterns. Predict Ratings.**

DineScope is a multi-task machine learning and data analytics project that studies restaurant data to identify patterns, predict ratings, recommend restaurants, classify cuisines, and analyse geographical distribution.

The project was developed as part of a Machine Learning Internship at **Cognifyz Technologies**. It contains four independently organised tasks, supported by Python scripts, executed verification notebooks, reports, visualisations, and trained model artifacts.

---

## Project Overview

DineScope is organised into four tasks.

### 1. Restaurant Rating Prediction

**Objective:** Predict a restaurant's aggregate rating using available restaurant attributes.

Work completed:
- Audited the dataset and performed exploratory data analysis.
- Cleaned and preprocessed the data.
- Trained and compared regression models.
- Evaluated model performance using MAE, RMSE, and R².
- Performed model diagnostics, feature-importance analysis, and hyperparameter tuning.
- Built and tested a prediction pipeline.

Final model results recorded in the verification analysis:

| Model configuration | RMSE | MAE | R² |
|---|---:|---:|---:|
| Selected model with Votes | 0.3330 | 0.2461 | 0.6414 |
| Selected model without Votes | 0.3860 | 0.2993 | 0.5183 |

The two configurations are reported separately because the availability of Votes changes the prediction feature set. These are experimental results on the project dataset, not guarantees of performance on other data.

### 2. Restaurant Recommendation System

**Objective:** Recommend restaurants using restaurant attributes and user preferences.

Work completed:
- Audited and prepared recommendation data.
- Engineered cuisine and restaurant features.
- Implemented a content-based recommendation approach.
- Combined cuisine similarity and rating information using an **85:15 cuisine-similarity/rating-score weighting**.
- Evaluated and tested recommendation behaviour.
- Created visualisations, reports, and an interactive recommendation interface.

### 3. Cuisine Classification

**Objective:** Predict cuisine labels from restaurant attributes using a multi-label classification approach.

Work completed:
- Audited and preprocessed cuisine data.
- Engineered classification features.
- Trained One-vs-Rest classification models using Logistic Regression and Linear Support Vector Classification.
- Tuned the prediction threshold and evaluated model behaviour.
- Built a cuisine prediction pipeline and interactive prediction interface.

Recorded threshold-tuning result:

| Evaluation | Micro F1 |
|---|---:|
| Baseline | 0.1545 |
| Tuned | 0.4124 |

The tuned result is the result recorded in the verification notebook. It should be interpreted in the context of the dataset, label distribution, and selected evaluation procedure.

### 4. Location-Based Analysis

**Objective:** Analyse restaurant distribution and identify location-level patterns and recommendation opportunities.

Work completed:
- Audited and prepared geographical data.
- Analysed restaurant distribution across cities and localities.
- Performed geospatial analysis using restaurant coordinates.
- Applied DBSCAN to explore spatial clusters.
- Conducted location-level statistical analysis.
- Developed weighted location recommendations and an interactive location explorer.

Recorded analysis summary:

| Analysis item | Result |
|---|---:|
| Restaurants in dataset | 9,551 |
| Restaurants with ratings | 7,403 |
| Restaurants without ratings | 2,148 |
| Usable coordinates | 9,054 |
| DBSCAN clusters | 172 |
| Records assigned to clusters | 7,463 |
| DBSCAN noise records | 1,591 |
| Eligible cities | 82 |
| Eligible localities | 176 |
| Statistical tests | 7 |
| Generated figures | 14 |

The recorded global city recommendation leader was **Orlando**; the top locality was **Connaught Place, New Delhi**. These rankings depend on the project's eligibility criteria and scoring method.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Data manipulation | Pandas, NumPy |
| Data visualisation | Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Model persistence | Joblib |
| Experiment tracking / ML tooling | MLflow |
| Interactive analysis | Jupyter Notebook, IPython |
| Development environment | VS Code |
| Version control | Git, GitHub |

---

## Repository Structure

```text
DineScope/
├── notebooks/
│   ├── 01_rating_prediction_verification.ipynb
│   ├── 02_recommendation_system_verification.ipynb
│   ├── 03_cuisine_classification_verification.ipynb
│   └── 04_location_based_analysis_verification.ipynb
├── src/
│   ├── task_1_rating_prediction/
│   ├── task_2_restaurant_recommendation/
│   ├── task_3_cuisine_classification/
│   └── task_4_location_analysis/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── task_1/
│   └── task_3/
├── reports/
│   ├── task_1/
│   ├── task_2/
│   ├── task_3/
│   └── task_4/
├── outputs/
│   └── figures/
│       ├── task_1/
│       ├── task_2/
│       ├── task_3/
│       └── task_4/
├── Machine Learning.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

**Data and model artifact handling:** The repository's `.gitignore` excludes raw data, processed data, and model binaries. The source code, notebooks, reports, figures, and project documentation are maintained separately from those artifacts. Obtain the dataset and any required artifacts through an authorised distribution and place them in the corresponding project directories before running scripts that depend on them.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Kamalesh-Suresh-Kumar/DineScope-Restaurant-Intelligence.git
cd DineScope-Restaurant-Intelligence
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Obtain the required data and artifacts

The raw dataset and generated processed/model artifacts are not tracked in this GitHub repository.

The expected raw dataset path is:

```text
data/raw/Dataset.csv
```

Task scripts may also depend on generated intermediate files and trained model artifacts. Obtain the authorised dataset and the corresponding task artifacts, then place them in the matching `data/processed/` and `models/` directories.

**Important:** Access to a dataset does not automatically grant permission to redistribute it. Follow the dataset provider's instructions and applicable licence terms. Do not redistribute the raw dataset, processed derivatives, trained models, or dataset-derived material unless the relevant permissions allow it.

### 5. Run the task workflows

Each task has its own numbered Python scripts. Run the scripts **in numerical order within the relevant task directory**, because later stages may consume files generated by earlier stages.

Example — Task 1:

```powershell
python src/task_1_rating_prediction/01_data_audit.py
python src/task_1_rating_prediction/02_exploratory_data_analysis.py
python src/task_1_rating_prediction/03_data_preprocessing.py
```

Continue with the remaining numbered scripts in that task directory.

The other task directories are:

```text
src/task_2_restaurant_recommendation/
src/task_3_cuisine_classification/
src/task_4_location_analysis/
```

Use the corresponding numbered scripts in each directory. Ensure the required raw data, intermediate files, and model artifacts are available before running a stage.

### 6. Review the verification notebooks

The `notebooks/` directory contains the four executed task verification notebooks. They preserve the analysis workflow and saved outputs used during project verification.

Open them with Jupyter:

```powershell
python -m jupyter notebook
```

---

## Reports and Visualisations

- `reports/` contains task-wise analysis reports and exported results.
- `outputs/figures/` contains task-wise visualisations.
- `notebooks/` contains the executed verification notebooks and their saved outputs.

These materials document the experiments and analysis performed for the project. Their interpretation should account for the limitations described below.

---

## Important Analytical Notes and Limitations

- The location analysis excluded **497 records with suspicious zero coordinates** from coordinate-based analysis.
- DBSCAN identifies density-based spatial groupings; cluster membership is not a measure of geographic area or service quality.
- Location density is based on restaurant counts and is not area-normalised.
- Mean coordinates are approximate geographic summaries.
- Statistical tests identify associations or differences under their assumptions; they do not establish causation.
- City and locality rankings are global results under the project's eligibility and scoring rules. They should not be interpreted as universal rankings.
- Rating predictions and cuisine classifications are constrained by the available dataset, feature quality, label distribution, and evaluation design.
- Recommendation results depend on the selected similarity and scoring strategy and on the information available for each restaurant.

---

## Project Status

- [x] Dataset auditing and exploratory data analysis
- [x] Restaurant rating prediction and model evaluation
- [x] Rating model diagnostics and hyperparameter tuning
- [x] Content-based restaurant recommendation system
- [x] Multi-label cuisine classification and threshold tuning
- [x] Geospatial and location-based analysis
- [x] Location-level statistical analysis
- [x] Interactive task workflows
- [x] Four executed task verification notebooks
- [x] Task-wise reports and visualisations
- [x] Final consolidated project report and demonstration materials

---

## Project Context

Developed as part of a **Machine Learning Internship at Cognifyz Technologies**.

## Disclaimer

DineScope is an educational and analytical project. Results are based on the supplied dataset and the methods implemented in this repository. Predictions, classifications, recommendations, statistical findings, and geographical rankings should be interpreted within the stated methodological and data limitations. They should not be treated as guarantees or as a substitute for independent verification.


# Author
<b>Name: Kamalesh S P</b><br/>
<b>Roll Number: 230701138</b><br/>
<b>Profolio: </b>[Portfolio link](https://kamalesh-suresh-kumar.github.io/Portfolio-KAI-HQ/)<br/>
<b>Connect with Me: </b>[Connection Link](https://kamalesh-suresh-kumar.github.io/KAI-Nexus/)<br/>