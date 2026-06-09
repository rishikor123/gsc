# 🍪 Smart Cookies: A Predictive Approach to Girl Scout Cookie Sales

> A full-stack web application that forecasts Girl Scout cookie demand at the **troop and cookie-type level** using a hybrid, multi-model machine learning system — replacing a legacy method that explained only ~10% of sales variability and unlocking an estimated **$1.09M** in value for Girl Scouts of Central Indiana.

**Presented at the INFORMS Analytics Conference and the Purdue Undergraduate Research Conference (2nd Place, Applied Analytics Impact).**

**Authors:** Rishita Korapati · Gabriel Morales · Ramya Chowdary Polineni
**Affiliation:** Purdue University, Mitchell E. Daniels, Jr. School of Business
**Partner:** Girl Scouts of Central Indiana · **Supported by:** Krenicki Center for Business Analytics & Machine Learning

---

## 📌 Overview

Every year, thousands of Girl Scouts rely on cookie sales for fundraising. The existing forecasting method — based solely on the **previous year's numbers** — explained only about **10% of sales variability**, frequently leading to **missed revenue (under-ordering)** or **excess stock (over-ordering)**.

This project builds a data-driven alternative: a **Hybrid Multi-Model** forecasting system, wrapped in a **Flask web app**, that integrates historical sales, troop participation, and seasonality to predict demand for **each troop and each cookie type** — automatically choosing the best-performing model per segment and serving the results through an interactive interface.

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

## 🖥️ The Application

A full-stack app that puts the forecasting model in the hands of non-technical users (troop leaders and council staff):

- **Backend — `app.py` (Flask):** loads `FinalCookieSales.csv`, runs the hybrid forecasting logic, and serves predictions to the frontend.
- **Frontend — `templates/index.html` + `static/index.css` + `frontend/manualapp.js`:** an interactive UI for exploring troop-level forecasts and sales trends.
- **Deployment — `Procfile`:** defines the web process for one-click deployment to platforms like Heroku or Render.

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

The app runs against **`FinalCookieSales.csv`**, included in the repository.

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


---

## 🧭 Methodology (SEMMA)

1. **Data Understanding** — entity-relationship mapping of troop sales; categorical/numerical variable profiling; initial exploration and cleaning.
2. **Data Preprocessing** — drop irrelevant columns, handle missing values, scale features with `StandardScaler`, apply transformations and feature engineering.
3. **Exploratory Data Analysis** — segment sales by troop; time-series decomposition to surface patterns; visual analysis of seasonality and trends.
4. **Modeling & Validation** — group by troop × cookie type; split by year into **train (2020–2023)** and **test (2024)**; evaluate Ridge, Random Forest, Polynomial, XGBoost, and Linear Regression; validate with an RMSE-based heuristic.
5. **Predictive Modeling** — dynamic selection of the best model per segment using RMSE, MSE, MAE, MAPE, and R².
6. **Reporting & Insights** — troop-level forecasts and next-cycle quantity predictions, surfaced through the web app.

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
- **Web framework:** Flask
- **ML / data:** pandas, NumPy, scikit-learn (Ridge, LinearRegression, RandomForest, Polynomial features), XGBoost
- **Preprocessing:** `StandardScaler`, cross-validation
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Gunicorn + `Procfile` (Heroku / Render-ready)

---

## 📁 Repository Structure

```
smart-cookies/
├── frontend/
│   └── manualapp.js        # frontend logic
├── static/
│   └── index.css           # styling
├── templates/
│   └── index.html          # main page (Flask template)
├── FinalCookieSales.csv    # sales dataset used by the app
├── app.py                  # Flask backend: forecasting + serving predictions
├── Procfile                # process definition for deployment (gunicorn)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/smart-cookies.git
cd smart-cookies

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app locally
python app.py                   # or: flask run  /  gunicorn app:app
```

Then open the app in your browser (Flask defaults to **http://127.0.0.1:5000**).

### Deployment
The `Procfile` defines the web process (e.g., `web: gunicorn app:app`), so the app deploys directly to platforms like **Heroku** or **Render**. Push the repo, set the build to install `requirements.txt`, and the platform runs the process from the `Procfile`.

---

## 🔄 Future Work

- **Improve data collection & quality** — maintain detailed sales histories for more precise forecasting and better generalization.
- **Incorporate external factors** — weather, marketing efforts, and regional trends to further refine accuracy.
- **Operationalize** — expand the app for council-wide troop-level forecasting.
- **Retrain regularly** — refresh with new data, gather user feedback, and adapt to changing business needs.

---

## 🙏 Acknowledgements

Thanks to **Professor Davi Moreira** for guidance and support throughout the project, to the **Girl Scouts of Central Indiana** for providing the data central to this work, and to the **Krenicki Center for Business Analytics & Machine Learning** for its support and resources.
