from flask import Flask, request, render_template, url_for, jsonify
from flask_cors import CORS
import pandas as pd
import statsmodels.api as sm
import numpy as np
import warnings
import time
import re

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

warnings.simplefilter("ignore", category=RuntimeWarning)

app = Flask(__name__)
CORS(app)

# ----------------------------------------------------------------
# LOAD AND PREPROCESS
# ----------------------------------------------------------------
df = pd.read_csv('FinalCookieSales.csv')
df = df.drop(columns=['date'], errors='ignore')
df['troop_id'] = pd.to_numeric(df['troop_id'], errors='coerce').astype('Int64')
df['period'] = pd.to_numeric(df['period'], errors='coerce').astype('Int64')
df['number_of_girls'] = pd.to_numeric(df['number_of_girls'], errors='coerce').astype(float)
df['number_cases_sold'] = pd.to_numeric(df['number_cases_sold'], errors='coerce')
df = df.dropna()
df = df[df['number_cases_sold'] > 0]
df['troop_id'] = df['troop_id'].astype(int)
df['period'] = df['period'].astype(int)
df['period_squared'] = df['period'] ** 2

historical_stats = df.groupby(['troop_id', 'cookie_type'])['number_cases_sold'].agg(['min', 'max']).reset_index()
historical_stats.columns = ['troop_id', 'cookie_type', 'historical_low', 'historical_high']
df = df.merge(historical_stats, on=['troop_id', 'cookie_type'], how='left')

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
def normalize_cookie_type(raw_name: str) -> str:
    raw_lower = raw_name.strip().lower()
    slug = re.sub(r'[^a-z0-9]+', '', raw_lower)
    return normalized_to_canonical.get(slug, raw_name)

df['canonical_cookie_type'] = df['cookie_type'].apply(normalize_cookie_type)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_ridge_interval_analysis():
    start_time = time.time()
    groups = df.groupby(['troop_id', 'canonical_cookie_type'])
    total_groups = len(groups)

    y_train_all = []
    y_train_pred_all = []
    y_test_all = []
    y_test_pred_all = []
    test_records = []

    ridge_alphas = np.logspace(-2, 3, 10)

    for (troop, cookie), group in tqdm(groups, total=total_groups, desc="Processing Models"):
        group = group.sort_values(by='period')
        train = group[group['period'] <= 4]
        test = group[group['period'] == 5]
        if train.empty or test.empty:
            continue

        X_train = train[['period', 'number_of_girls']]
        y_train = train['number_cases_sold']
        X_test = test[['period', 'number_of_girls']]
        y_test = test['number_cases_sold']

        if X_train.empty:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        best_model = None
        best_mae = float('inf')
        best_y_train_pred = None
        best_y_test_pred = None

        for alpha in ridge_alphas:
            try:
                model = Ridge(alpha=alpha)
                model.fit(X_train_scaled, y_train)
                y_train_pred_ = model.predict(X_train_scaled)
                y_test_pred_ = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_test_pred_)
                if mae < best_mae:
                    best_mae = mae
                    best_model = model
                    best_y_train_pred = y_train_pred_
                    best_y_test_pred = y_test_pred_
            except:
                continue

        if best_model is not None:
            y_train_all.extend(y_train)
            y_train_pred_all.extend(best_y_train_pred)
            y_test_all.extend(y_test)
            y_test_pred_all.extend(best_y_test_pred)

            tdf = test.copy()
            tdf['troop_id'] = troop
            tdf['canonical_cookie_type'] = cookie
            tdf['predicted'] = best_y_test_pred
            test_records.append(tdf)

    y_train_all = np.array(y_train_all)
    y_train_pred_all = np.array(y_train_pred_all)
    y_test_all = np.array(y_test_all)
    y_test_pred_all = np.array(y_test_pred_all)

    mae_train = mean_absolute_error(y_train_all, y_train_pred_all)
    mse_train = mean_squared_error(y_train_all, y_train_pred_all)
    rmse_train = np.sqrt(mse_train)

    app.config['OVERALL_RIDGE_RMSE'] = rmse_train
    end_time = time.time()
    print(f"Ridge analysis done in {end_time - start_time:.2f} seconds.")

run_ridge_interval_analysis()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/troop_ids', methods=['GET'])
def get_troop_ids():
    troop_ids = sorted(df['troop_id'].unique().tolist())
    return jsonify(troop_ids)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    chosen_troop = data.get('troop_id')
    chosen_num_girls = data.get('num_girls')
    chosen_period = data.get('year', 5)

    try:
        chosen_troop = int(chosen_troop)
        chosen_num_girls = float(chosen_num_girls)
        chosen_period = int(chosen_period)
    except ValueError:
        return jsonify({"error": "Invalid input."}), 400

    if chosen_num_girls == 0:
        return jsonify([])

    df_troop = df[(df['troop_id'] == chosen_troop) & (df['period'] < chosen_period)].copy()
    if df_troop.empty:
        return jsonify([])

    overall_ridge_rmse = app.config.get('OVERALL_RIDGE_RMSE', 0.0)
    coverage_factor = 2.0
    interval_width = coverage_factor * overall_ridge_rmse

    predictions = []
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

    df_troop['canonical_cookie_type'] = df_troop['cookie_type'].apply(normalize_cookie_type)

    for cookie_type, group in df_troop.groupby('canonical_cookie_type'):
        if group['period'].nunique() < 2:
            last_period = group['period'].max()
            last_val = group.loc[group['period'] == last_period, 'number_cases_sold'].mean()
            interval_lower = max(last_val - interval_width, 1)
            interval_upper = last_val + interval_width
            img_file = cookie_images.get(cookie_type, 'default.png')
            img_url = url_for('static', filename=img_file, _external=True)
            predictions.append({
                "cookie_type": cookie_type,
                "predicted_cases": round(last_val, 2),
                "interval_lower": round(interval_lower, 2),
                "interval_upper": round(interval_upper, 2),
                "image_url": img_url
            })
            continue

        X_train = group[['period', 'period_squared', 'number_of_girls']]
        y_train = group['number_cases_sold']
        X_train = sm.add_constant(X_train)

        try:
            model = sm.OLS(y_train, X_train).fit()
            period_squared = chosen_period ** 2
            X_test = np.array([[1, chosen_period, period_squared, chosen_num_girls]])
            predicted_cases = model.predict(X_test)[0]

            historical_low = group['historical_low'].iloc[0]
            historical_high = group['historical_high'].iloc[0]
            predicted_cases = max(historical_low, min(predicted_cases, historical_high))

            interval_lower = max(predicted_cases - interval_width, 1)
            interval_upper = predicted_cases + interval_width

            img_file = cookie_images.get(cookie_type, 'default.png')
            img_url = url_for('static', filename=img_file, _external=True)

            predictions.append({
                "cookie_type": cookie_type,
                "predicted_cases": round(predicted_cases, 2),
                "interval_lower": round(interval_lower, 2),
                "interval_upper": round(interval_upper, 2),
                "image_url": img_url
            })
        except Exception as e:
            print(f"Error in OLS for {cookie_type}: {e}")

    return jsonify(predictions)

@app.route('/api/history/<int:troop_id>', methods=['GET'])
def get_troop_history(troop_id):
    troop_df = df[df['troop_id'] == troop_id].copy()
    if troop_df.empty:
        return jsonify({"error": "No data found"}), 404

    grouped_sales = troop_df.groupby('period')['number_cases_sold'].sum().reset_index()
    total_sales = []
    for _, row in grouped_sales.iterrows():
        total_sales.append({
            "period": int(row['period']),
            "totalSales": float(row['number_cases_sold'])
        })

    grouped_girls = troop_df.groupby('period')['number_of_girls'].mean().reset_index()
    girls_data = []
    for _, row in grouped_girls.iterrows():
        girls_data.append({
            "period": int(row['period']),
            "numberOfGirls": float(row['number_of_girls'])
        })

    return jsonify({
        "totalSalesByPeriod": total_sales,
        "girlsByPeriod": girls_data
    })

@app.route('/api/cookie_breakdown/<int:troop_id>', methods=['GET'])
def get_cookie_breakdown(troop_id):
    troop_df = df[df['troop_id'] == troop_id].copy()
    if troop_df.empty:
        return jsonify([])

    grouped = troop_df.groupby(['period', 'canonical_cookie_type'])['number_cases_sold'].sum().reset_index()
    pivoted = grouped.pivot(index='period', columns='canonical_cookie_type', values='number_cases_sold').fillna(0)
    pivoted.reset_index(inplace=True)

    data_list = pivoted.to_dict(orient='records')

    return jsonify(data_list)

if __name__ == '__main__':
    app.run(debug=True)
