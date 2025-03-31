from flask import Flask, request, jsonify, url_for, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
import time
import warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

warnings.simplefilter("ignore", category=RuntimeWarning)

app = Flask(__name__, static_folder="static")
CORS(app)

# -------------------------------
# DATA LOADING & PREPROCESSING
# -------------------------------
df = pd.read_csv('FinalCookieSales.csv')
df = df.drop(columns=['date'], errors='ignore')
df = df.dropna()
df = df[df['number_cases_sold'] > 0]

df['troop_id'] = df['troop_id'].astype(int)
df['period'] = df['period'].astype(int)
df['number_of_girls'] = df['number_of_girls'].astype(float)
df['number_cases_sold'] = df['number_cases_sold'].astype(float)
df['period_squared'] = df['period'] ** 2

# Normalize cookie types
normalized_to_canonical = {
    'adventurefuls': 'Adventurefuls',
    'dosidos': 'Do-Si-Dos',
    'samoas': 'Samoas',
    'smores': "S'mores",
    'tagalongs': 'Tagalongs',
    'thinmints': 'Thin Mints',
    'toffeetastic': 'Toffee-tastic',
    'trefoils': 'Trefoils',
    'lemonups': 'Lemon-Ups'
}

def normalize_cookie_type(raw_name):
    raw_lower = raw_name.strip().lower()
    slug = re.sub(r'[^a-z0-9]+', '', raw_lower)
    return normalized_to_canonical.get(slug, raw_name)

df['canonical_cookie_type'] = df['cookie_type'].apply(normalize_cookie_type)

# Add historical stats for interval clamping
stats = df.groupby(['troop_id', 'canonical_cookie_type'])['number_cases_sold'].agg(['min', 'max']).reset_index()
stats.columns = ['troop_id', 'canonical_cookie_type', 'historical_low', 'historical_high']
df = df.merge(stats, on=['troop_id', 'canonical_cookie_type'], how='left')

# -------------------------------
# TRAIN RIDGE TO GET RMSE FOR INTERVAL WIDTH
# -------------------------------
def run_ridge_interval_analysis():
    groups = df.groupby(['troop_id', 'canonical_cookie_type'])
    y_train_all, y_pred_all = [], []

    for (troop, cookie), group in tqdm(groups):
        group = group.sort_values('period')
        train = group[group['period'] <= 4]
        test = group[group['period'] == 5]
        if train.empty or test.empty:
            continue

        X_train = train[['period', 'number_of_girls']]
        y_train = train['number_cases_sold']
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        best_model = Ridge(alpha=1.0)
        best_model.fit(X_train_scaled, y_train)
        y_pred = best_model.predict(X_train_scaled)

        y_train_all.extend(y_train)
        y_pred_all.extend(y_pred)

    rmse = np.sqrt(mean_squared_error(y_train_all, y_pred_all))
    app.config['OVERALL_RIDGE_RMSE'] = rmse
    print(f"Global RMSE for prediction interval: {rmse:.2f}")

run_ridge_interval_analysis()

# -------------------------------
# API ROUTES
# -------------------------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/api/troop_ids')
def get_troop_ids():
    return jsonify(sorted(df['troop_id'].unique().tolist()))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    troop_id = int(data.get('troop_id'))
    num_girls = float(data.get('num_girls'))
    period = int(data.get('year', 5))

    df_troop = df[(df['troop_id'] == troop_id) & (df['period'] < period)].copy()
    if df_troop.empty:
        return jsonify([])

    interval_width = app.config['OVERALL_RIDGE_RMSE'] * 2
    df_troop['canonical_cookie_type'] = df_troop['cookie_type'].apply(normalize_cookie_type)

    cookie_images = {
        "Adventurefuls": "ADVEN.png",
        "Do-Si-Dos": "DOSI.png",
        "Samoas": "SAM.png",
        "S'mores": "SMORE.png",
        "Tagalongs": "TAG.png",
        "Thin Mints": "THIN.png",
        "Toffee-tastic": "TFTAS.png",
        "Trefoils": "TREF.png",
        "Lemon-Ups": "LMNUP.png"
    }

    predictions = []
    for cookie_type, group in df_troop.groupby('canonical_cookie_type'):
        X = group[['period', 'period_squared', 'number_of_girls']]
        y = group['number_cases_sold']
        X = sm.add_constant(X)
        try:
            model = sm.OLS(y, X).fit()
            test_row = np.array([[1, period, period**2, num_girls]])
            pred = model.predict(test_row)[0]
        except Exception as e:
            print(f"Prediction error for {cookie_type}: {e}")
            continue

        hist_low = group['historical_low'].iloc[0]
        hist_high = group['historical_high'].iloc[0]
        pred = max(hist_low, min(pred, hist_high))

        image = cookie_images.get(cookie_type, "default.png")
        image_url = url_for('static', filename=image, _external=True)

        predictions.append({
            "cookie_type": cookie_type,
            "predicted_cases": round(pred, 2),
            "interval_lower": round(max(1, pred - interval_width), 2),
            "interval_upper": round(pred + interval_width, 2),
            "image_url": image_url
        })

    return jsonify(predictions)

@app.route('/api/history/<int:troop_id>')
def get_history(troop_id):
    troop_df = df[df['troop_id'] == troop_id]
    if troop_df.empty:
        return jsonify({"error": "No data"}), 404

    sales = troop_df.groupby('period')['number_cases_sold'].sum().reset_index()
    girls = troop_df.groupby('period')['number_of_girls'].mean().reset_index()

    return jsonify({
        "totalSalesByPeriod": [{"period": int(r['period']), "totalSales": r['number_cases_sold']} for _, r in sales.iterrows()],
        "girlsByPeriod": [{"period": int(r['period']), "numberOfGirls": r['number_of_girls']} for _, r in girls.iterrows()]
    })

@app.route('/api/cookie_breakdown/<int:troop_id>')
def get_breakdown(troop_id):
    troop_df = df[df['troop_id'] == troop_id]
    if troop_df.empty:
        return jsonify([])

    grouped = troop_df.groupby(['period', 'canonical_cookie_type'])['number_cases_sold'].sum().reset_index()
    pivoted = grouped.pivot(index='period', columns='canonical_cookie_type', values='number_cases_sold').fillna(0)
    pivoted.reset_index(inplace=True)

    return jsonify(pivoted.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
