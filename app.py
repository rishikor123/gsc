from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
import pandas as pd
import statsmodels.api as sm
import numpy as np
import warnings
import time
import re
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

warnings.simplefilter("ignore", category=RuntimeWarning)

app = Flask(__name__)
CORS(app)

# ----------------------------
# LOAD & CLEAN DATA
# ----------------------------
df = pd.read_csv('FinalCookieSales.csv')
df = df.drop(columns=['date'], errors='ignore')
df['troop_id'] = pd.to_numeric(df['troop_id'], errors='coerce').astype('Int64')
df['period'] = pd.to_numeric(df['period'], errors='coerce').astype('Int64')
df['number_of_girls'] = pd.to_numeric(df['number_of_girls'], errors='coerce')
df['number_cases_sold'] = pd.to_numeric(df['number_cases_sold'], errors='coerce')
df = df.dropna()
df = df[df['number_cases_sold'] > 0]
df['period_squared'] = df['period'] ** 2

normalized_to_canonical = {
    'adventurefuls': 'Adventurefuls', 'dosidos': 'Do-Si-Dos', 'samoas': 'Samoas',
    'smores': "S'mores", 'tagalongs': 'Tagalongs', 'thinmints': 'Thin Mints',
    'toffeetastic': 'Toffee-Tastic', 'trefoils': 'Trefoils', 'lemonups': 'Lemon-Ups'
}

def normalize_cookie_type(raw_name: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '', raw_name.lower().strip())
    return normalized_to_canonical.get(slug, raw_name)

df['canonical_cookie_type'] = df['cookie_type'].apply(normalize_cookie_type)

hist_stats = df.groupby(['troop_id', 'canonical_cookie_type'])['number_cases_sold'].agg(['min', 'max']).reset_index()
hist_stats.columns = ['troop_id', 'canonical_cookie_type', 'historical_low', 'historical_high']
df = df.merge(hist_stats, on=['troop_id', 'canonical_cookie_type'], how='left')

# ----------------------------
# TRAIN RIDGE MODELS
# ----------------------------
def run_ridge_analysis():
    rmse_total = []
    groups = df.groupby(['troop_id', 'canonical_cookie_type'])
    ridge_alphas = np.logspace(-2, 3, 10)

    for (_, _), group in tqdm(groups, desc="Ridge Training"):
        train = group[group['period'] <= 4]
        test = group[group['period'] == 5]
        if train.empty or test.empty:
            continue

        X_train = train[['period', 'number_of_girls']]
        y_train = train['number_cases_sold']
        X_test = test[['period', 'number_of_girls']]
        y_test = test['number_cases_sold']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        best_rmse = float('inf')
        for alpha in ridge_alphas:
            model = Ridge(alpha=alpha)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            if rmse < best_rmse:
                best_rmse = rmse

        rmse_total.append(best_rmse)

    app.config['RIDGE_RMSE'] = np.mean(rmse_total)

run_ridge_analysis()

# ----------------------------
# API ROUTES
# ----------------------------
@app.route('/api/troop_ids')
def get_troop_ids():
    return jsonify(sorted(df['troop_id'].dropna().unique().astype(int).tolist()))

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    troop_id = int(data.get('troop_id'))
    num_girls = float(data.get('num_girls'))
    period = int(data.get('year', 5))

    troop_df = df[(df['troop_id'] == troop_id) & (df['period'] < period)].copy()
    if troop_df.empty:
        return jsonify([])

    interval = app.config.get('RIDGE_RMSE', 10) * 2
    results = []

    cookie_images = {
        "Adventurefuls": "ADVEN.png",
        "Do-Si-Dos": "DOSI.png",
        "Lemon-Ups": "LMNUP.png",
        "Samoas": "SAM.png",
        "Tagalongs": "TAG.png",
        "Thin Mints": "THIN.png",
        "Toffee-Tastic": "TFTAS.png",
        "Trefoils": "TREF.png",
        "S'mores": "SMORE.png"
    }

    for cookie_type, group in troop_df.groupby('canonical_cookie_type'):
        X = group[['period', 'period_squared', 'number_of_girls']]
        y = group['number_cases_sold']
        X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X).fit()
            pred_input = sm.add_constant(pd.DataFrame([{
                'period': period,
                'period_squared': period ** 2,
                'number_of_girls': num_girls
            }]))
            pred = model.predict(pred_input)[0]

            low = group['historical_low'].iloc[0]
            high = group['historical_high'].iloc[0]
            pred = np.clip(pred, low, high)
            lower, upper = max(pred - interval, 1), pred + interval

            image_file = cookie_images.get(cookie_type, 'default.png')
            image_url = url_for('static', filename=image_file, _external=True)

            results.append({
                'cookie_type': cookie_type,
                'predicted_cases': round(pred, 2),
                'interval_lower': round(lower, 2),
                'interval_upper': round(upper, 2),
                'image_url': image_url
            })
        except Exception as e:
            print(f"OLS error for {cookie_type}: {e}")

    return jsonify(results)

@app.route('/api/history/<int:troop_id>')
def history(troop_id):
    troop_df = df[df['troop_id'] == troop_id]
    if troop_df.empty:
        return jsonify({"error": "No data found"}), 404

    total_sales = troop_df.groupby('period')['number_cases_sold'].sum().reset_index()
    total_sales.columns = ['period', 'totalSales']
    girls = troop_df.groupby('period')['number_of_girls'].mean().reset_index()
    girls.columns = ['period', 'numberOfGirls']

    return jsonify({
        "totalSalesByPeriod": total_sales.to_dict(orient='records'),
        "girlsByPeriod": girls.to_dict(orient='records')
    })

@app.route('/api/cookie_breakdown/<int:troop_id>')
def breakdown(troop_id):
    troop_df = df[df['troop_id'] == troop_id]
    if troop_df.empty:
        return jsonify([])

    pivot = troop_df.groupby(['period', 'canonical_cookie_type'])['number_cases_sold'].sum().unstack().fillna(0).reset_index()
    return jsonify(pivot.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
