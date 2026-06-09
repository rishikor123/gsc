# 🍪 Smart Cookies: A Predictive Approach to Girl Scout Cookie Sales

> A hybrid, multi-model machine learning system that forecasts Girl Scout cookie demand at the **troop and cookie-type level**, replacing a legacy method that explained only ~10% of sales variability — and unlocking an estimated **$1.09M** in value for Girl Scouts of Central Indiana.

**Presented at the INFORMS Analytics Conference and the Purdue Undergraduate Research Conference (2nd Place, Applied Analytics Impact).**

**Authors:** Rishita Korapati · Gabriel Morales · Ramya Chowdary Polineni
**Affiliation:** Purdue University, Mitchell E. Daniels, Jr. School of Business
**Partner:** Girl Scouts of Central Indiana · **Supported by:** Krenicki Center for Business Analytics & Machine Learning

---

## 📌 Overview

Every year, thousands of Girl Scouts rely on cookie sales for fundraising. The existing forecasting method — based solely on the **previous year's numbers** — explained only about **10% of sales variability**, frequently leading to **missed revenue (under-ordering)** or **excess stock (over-ordering)**.

This project builds a data-driven alternative: a **Hybrid Multi-Model** forecasting system that integrates historical sales, troop participation, and seasonality to predict demand for **each troop and each cookie type**, automatically choosing the best-performing model per segment.

### Key Results

| Metric | Value |
|---|---|
| Forecast error (RMSE) | reduced from **~14 → ~10** |
| MAE | ~4.55 |
| MAPE | ~5.7% |
| R² | ~0.87 |
| Additional boxes correctly forecasted | **181,000+** |
| Estimated value generated | **$1,089,000+** |
| Scale | **1,401 troops · 8 cookie types · 12 boxes/case** |

---

## ❓ Research Questions

1. Can machine learning models integrate historical sales data and troop participation rates to **improve cookie-sales forecast accuracy** beyond traditional methods?
2. How can predictive insights be used to **optimize inventory and marketing** for individual troops — ensuring each meets demand without costly surplus or shortage?

---

## 🎯 Business Impact

- **Inventory management & cost savings** — fewer over- and under-orders at the troop level.
- **Enhanced marketing strategies** — demand signals by cookie type and region.
- **Increased troop engagement** — clearer, achievable targets.
- **Data-driven decision making** — replacing intuition with evidence.

By improving prediction accuracy per troop–cookie pair over the legacy SIO tool and scaling across 1,401 troops, 8 cookie types, and 12 boxes per case, the model translates to a potential impact of **over 181,000 boxes — more than $1.09M in value** — enabling GS Indiana to optimize inventory, reduce over-ordering, and boost revenue.

---

## 🗂️ Dataset

| Property | Detail |
|---|---|
| Rows | 68,966 |
| Columns | 6 |
| Coverage | Multiple sales periods across troops with varying participation rates |

**Key variables**

| Variable | Description |
|---|---|
| `Date` | Sales transaction date |
| `Number of Cases Sold` | Total boxes sold |
| `Cookie Type` | Cookie variety |
| `Troop ID` | Troop responsible for the sale |
| `Number of Girls` | Girl Scouts participating in sales |
| `Period` | Specific sales time window |

**Data quality**
- **Outliers:** extreme cases removed after validation.
- **Missing values:** minimal; handled through imputation.

> ⚠️ The underlying sales data belongs to Girl Scouts of Central Indiana and is **not** included in this repository. See [Data Access](#-data-access) below.

---

## 🧭 Methodology (SEMMA)

1. **Data Understanding** — entity-relationship mapping of troop sales; categorical/numerical variable profiling; initial exploration and cleaning.
2. **Data Preprocessing** — drop irrelevant columns, handle missing values, scale features with `StandardScaler`, apply transformations and feature engineering.
3. **Exploratory Data Analysis** — segment sales by troop; time-series decomposition to surface patterns; visual analysis of seasonality and trends.
4. **Modeling & Validation** — group by troop × cookie type; split by year into **train (2020–2023)** and **test (2024)**; evaluate Ridge, Random Forest, Polynomial, XGBoost, and Linear Regression; validate with an RMSE-based heuristic.
5. **Predictive Modeling** — dynamic selection of the best model per segment using RMSE, MSE, MAE, MAPE, and R².
6. **Reporting & Insights** — troop-level forecasts, next-cycle quantity predictions, and a recommendations report (plus a demo for sales-trend exploration).

**Language:** Python

---

## 🤖 Modeling Approach

### Baselines
- **SIO Model** — uses last year's sales and troop participation to estimate the next year.
- **Average Model** — averages 2021–2023 sales to predict 2024.

Both baselines produced **higher RMSE**, indicating significant prediction error and motivating a more adaptive approach.

### Hybrid Multi-Model System
The final solution **automatically selects the best method per troop–cookie pair** from:
- Clustered Ridge Regression
- Troop-Level Ridge Regression
- Location-Level Ridge Regression
- Linear Regression
- SIO & Average baselines

Each prediction uses the method with the **lowest expected error**, chosen dynamically from past performance. Cluster-based modeling within each location and cookie type enables **personalized, segment-aware forecasts**.

### Validation
- **Cross-validation** within each training group to tune the regularization strength (λ), minimizing overfitting and stabilizing performance across troops and cookie types.
- **Dynamic error-based selection** — for each troop–cookie pair, pick the method with the lowest MSE on historical performance.
- **Robustness across segments** — the clustering → Ridge → fallback-heuristic design adapts to sparse, dense, and noisy troop histories, delivering consistent gains over baselines.

### Assumptions & Limitations
- Assumes past sales patterns are predictive of future performance, with no major disruptions to cookie availability, troop operations, or supply chains.
- May underperform for troops with **limited or erratic historical data**.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Core libraries:** pandas, NumPy, scikit-learn (Ridge, LinearRegression, RandomForest, Polynomial features), XGBoost
- **Preprocessing:** `StandardScaler`, cross-validation
- **Visualization / reporting:** Matplotlib / Seaborn, plus a sales-trend demo

> _Update this list to match the exact libraries imported in the notebook._

---

## 📁 Repository Structure

```
smart-cookies/
├── data/                 # (gitignored) raw + processed data — not committed
├── notebooks/            # exploration, EDA, and modeling notebooks
├── src/                  # reusable code (preprocessing, models, evaluation)
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluate.py
├── reports/              # figures, results, and the conference poster
│   └── smart_cookies_poster.png
├── requirements.txt
└── README.md
```

> _Adjust to match your actual file layout once you commit the code._

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/smart-cookies.git
cd smart-cookies

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the analysis
jupyter notebook notebooks/
```

> _TODO: replace with your real run commands / entry point once the code is in the repo._

---

## 🔒 Data Access

The sales dataset is proprietary to **Girl Scouts of Central Indiana** and is not distributed here. The code is structured to run against a CSV with the schema described in [Dataset](#-dataset). To reproduce results, supply your own data matching that schema in `data/`.

---

## 🔄 Model Lifecycle & Future Work

- **Improve data collection & quality** — maintain detailed sales histories for more precise forecasting and better generalization.
- **Incorporate external factors** — weather, marketing efforts, and regional trends to further refine accuracy.
- **Operationalize** — use for troop-level forecasting to minimize error and maximize predictive accuracy.
- **Retrain regularly** — refresh with new data, gather user feedback, and adapt to changing business needs.

---

## 🙏 Acknowledgements

Thanks to **Professor Davi Moreira** for guidance and support throughout the project, to the **Girl Scouts of Central Indiana** for providing the data central to this work, and to the **Krenicki Center for Business Analytics & Machine Learning** for its support and resources.

---

## 📄 License

_Choose a license (e.g., MIT) for the code. Note that the dataset is not covered by this license and remains the property of Girl Scouts of Central Indiana._
