#  Model Comparison CLI — Petra Telecom Churn Prediction

A production-ready command-line interface (CLI) for training, evaluating, and comparing multiple machine learning models for customer churn prediction at Petra Telecom.

This project transforms an experimental notebook into a **reproducible, production-grade ML pipeline** suitable for real-world deployment workflows.

---

##  Overview

This system allows you to:

* Load and validate telecom churn datasets
* Train multiple machine learning models
* Evaluate models using stratified cross-validation
* Compare performance using multiple metrics (Accuracy, Precision, Recall, F1, PR-AUC)
* Perform threshold optimization for deployment decisions
* Generate plots and reports automatically
* Save all outputs in a structured directory

---

##  Installation

### 1. Create virtual environment (recommended)

```bash
python -m venv .venv
```

### 2. Activate environment

```bash
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Usage

###  Full Pipeline Execution

Runs complete training, evaluation, and reporting pipeline.

```bash
python compare_models.py --data-path data/telecom_churn.csv
```

###  Dry Run (Validation Mode)

Validates dataset and prints pipeline configuration without training models.

```bash
python compare_models.py --data-path data/telecom_churn.csv --dry-run
```

---

##  CLI Arguments

| Argument        | Type     | Default    | Description                       |
| --------------- | -------- | ---------- | --------------------------------- |
| `--data-path`   | Required | None       | Path to input CSV dataset         |
| `--output-dir`  | Optional | `./output` | Directory to store outputs        |
| `--n-folds`     | Optional | `5`        | Number of cross-validation folds  |
| `--random-seed` | Optional | `42`       | Random seed for reproducibility   |
| `--dry-run`     | Flag     | False      | Run validation only (no training) |

---

##  Outputs

All generated artifacts are saved to the output directory:

```
output/
│
├── comparison_table.csv
├── experiment_log.csv
├── pr_curves.png
├── calibration.png
├── threshold_sweep.png
├── threshold_sweep.csv
├── best_model.joblib
```

---

##  Example Commands

### Full execution

```bash
python compare_models.py --data-path data/telecom_churn.csv --output-dir output
```

### Dry run validation

```bash
python compare_models.py --data-path data/telecom_churn.csv --dry-run
```

### Custom configuration

```bash
python compare_models.py --data-path data/telecom_churn.csv --n-folds 10 --random-seed 7
```

---

##  Key Features

* Production-ready CLI architecture (argparse-based)
* Structured logging instead of print statements
* Fully reproducible experiments (seed control)
* Cross-validation model benchmarking
* Business-aware threshold optimization
* Automatic artifact generation
* Modular and extensible design

---

##  Project Structure

```
.
├── compare_models.py
├── data/
│   └── telecom_churn.csv
├── output/
├── requirements.txt
└── README.md
```

---

##  Business Context

This pipeline is designed for telecom churn prediction where:

* Positive class = customer churn
* Objective = maximize recall while controlling contact cost
* Metric priority = PR-AUC over accuracy (due to class imbalance)

---

##  Author

- Hosam Alkhawaldeh
---

