from flask import Flask, request, jsonify
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

# ----------------------------------------------------------------
# 1. LOAD AND PREPROCESS DATASET ONCE AT STARTUP
# ----------------------------------------------------------------
df = pd.read_csv('FinalCookieSales.csv')

# Drop 'date' if it exists
df = df.drop(columns=['date'], errors='ignore')

# Convert numeric columns carefully
df['troop_id'] = pd.to_numeric(df['troop_id'], errors='coerce').astype('Int64')
df['period'] = pd.to_numeric(df['period'], errors='coerce').astype('Int64')
df['number_of_girls'] = pd.to_numeric(df['number_of_girls'], errors='coerce').astype(float)
df['number_cases_sold'] = pd.to_numeric(df['number_cases_sold'], errors='coerce')

df = df.dropna()
df = df[df['number_cases_sold'] > 0]

# Convert to Python ints
df['troop_id'] = df['troop_id'].astype(int)
df['period'] = df['period'].astype(int)

# Create squared period column for OLS
df['period_squared'] = df['period'] ** 2

# Calculate historical guardrails
historical_stats = df.groupby(['troop_id', 'cookie_type'])['number_cases_sold'].agg(['min', 'max']).reset_index()
historical_stats.columns = ['troop_id', 'cookie_type', 'historical_low', 'historical_high']
df = df.merge(historical_stats, on=['troop_id', 'cookie_type'], how='left')

# ----------------------------------------------------------------
# 2. NORMALIZATION & MAPPING FOR COOKIE TYPES
# ----------------------------------------------------------------
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
    # Remove everything except letters and digits
    slug = re.sub(r'[^a-z0-9]+', '', raw_lower)
    # Map slug to a canonical name if in the dict
    return normalized_to_canonical.get(slug, raw_name)

df['canonical_cookie_type'] = df['cookie_type'].apply(normalize_cookie_type)

# ----------------------------------------------------------------
# 3. RIDGE + INTERVAL ANALYSIS FUNCTION
# ----------------------------------------------------------------
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

        # Example: train on periods <=4, test on period=5
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

    # Convert to arrays
    y_train_all = np.array(y_train_all)
    y_train_pred_all = np.array(y_train_pred_all)
    y_test_all = np.array(y_test_all)
    y_test_pred_all = np.array(y_test_pred_all)

    # Compute overall training metrics
    mae_train = mean_absolute_error(y_train_all, y_train_pred_all)
    mse_train = mean_squared_error(y_train_all, y_train_pred_all)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train_all, y_train_pred_all)
    mape_train = mean_absolute_percentage_error(y_train_all, y_train_pred_all)

    # Compute overall testing metrics
    mae_test = mean_absolute_error(y_test_all, y_test_pred_all)
    mse_test = mean_squared_error(y_test_all, y_test_pred_all)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test_all, y_test_pred_all)
    mape_test = mean_absolute_percentage_error(y_test_all, y_test_pred_all)

    if test_records:
        test_df_all = pd.concat(test_records, ignore_index=True)
    else:
        test_df_all = pd.DataFrame()

    coverage_factor = 2.0
    overall_train_rmse = rmse_train
    interval_width = coverage_factor * overall_train_rmse

    coverage_rate = 0
    if not test_df_all.empty:
        test_df_all['interval_lower'] = test_df_all['predicted'] - interval_width
        test_df_all['interval_upper'] = test_df_all['predicted'] + interval_width
        test_df_all['in_interval'] = (
            (test_df_all['number_cases_sold'] >= test_df_all['interval_lower']) &
            (test_df_all['number_cases_sold'] <= test_df_all['interval_upper'])
        )
        test_df_all['error'] = np.abs(test_df_all['number_cases_sold'] - test_df_all['predicted'])
        coverage_rate = test_df_all['in_interval'].mean() * 100

    app.config['OVERALL_RIDGE_RMSE'] = overall_train_rmse

    print("\n--- Ridge + Interval Coverage Results ---")
    print("Training Set Metrics:")
    print(f"MAE: {mae_train:.2f}, MSE: {mse_train:.2f}, RMSE: {rmse_train:.2f}, "
          f"MAPE: {mape_train:.2f}%, R²: {r2_train:.4f}")
    print("\nTesting Set Metrics:")
    print(f"MAE: {mae_test:.2f}, MSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, "
          f"MAPE: {mape_test:.2f}%, R²: {r2_test:.4f}")
    print(f"\nOverall Training RMSE: {overall_train_rmse:.2f}")
    print(f"Interval Width: ±({coverage_factor} × {overall_train_rmse:.2f}) = ±{interval_width:.2f}")
    print(f"Overall Test Prediction Interval Coverage: {coverage_rate:.2f}%")

    if not test_df_all.empty:
        worst_preds = test_df_all.sort_values('error', ascending=False).head(10)
        print("\nTop 10 Worst Predictions (By Abs Error):")
        print(worst_preds[['troop_id', 'canonical_cookie_type', 'period', 'number_of_girls',
                           'number_cases_sold', 'predicted', 'interval_lower',
                           'interval_upper', 'in_interval', 'error']])
    else:
        print("\nNo test data found during Ridge analysis.")

    end_time = time.time()
    print(f"\nProcess completed in {end_time - start_time:.2f} seconds.\n")

# Run the analysis once at startup
run_ridge_interval_analysis()

# ----------------------------------------------------------------
# 4. SIMPLE ROOT ROUTE (Optional)
# ----------------------------------------------------------------
@app.route('/', methods=['GET'])
def home():
    return "Cookie Sales Predictor API is running!"

# ----------------------------------------------------------------
# 5. MAIN PREDICTION ROUTE (JSON)
# ----------------------------------------------------------------
@app.route('/api/predict', methods=['POST'])
def predict_api():
    """
    Expects JSON like:
      {
        "troop_id": 123,
        "num_girls": 20,
        "year": 5
      }
    Returns a JSON list of predictions for each cookie type.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided."}), 400

    # Extract parameters from JSON
    chosen_troop = data.get('troop_id')
    chosen_num_girls = data.get('num_girls')
    chosen_year = data.get('year', 5)  # default to period=5 if not provided

    # Validate
    try:
        chosen_troop = int(chosen_troop)
        chosen_num_girls = float(chosen_num_girls)
        chosen_period = int(chosen_year)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input. Must be numeric."}), 400

    # If no girls, trivial prediction
    if chosen_num_girls == 0:
        return jsonify({
            "message": f"Troop {chosen_troop}, Period {chosen_period}: zero girls => no cookies sold."
        })

    # Filter historical data: periods < chosen_period
    df_troop = df[(df['troop_id'] == chosen_troop) & (df['period'] < chosen_period)]
    if df_troop.empty:
        return jsonify({
            "error": f"No historical data for troop {chosen_troop} before period {chosen_period}."
        }), 404

    # We'll use the overall RMSE from the ridge analysis as interval width
    overall_ridge_rmse = app.config.get('OVERALL_RIDGE_RMSE', 0.0)
    coverage_factor = 2.0
    interval_width = coverage_factor * overall_ridge_rmse

    predictions = []

    # Normalize cookie types in df_troop to match the canonical name
    df_troop['canonical_cookie_type'] = df_troop['cookie_type'].apply(normalize_cookie_type)

    # For each cookie type in the troop's historical data, fit an OLS model
    for cookie_type, group in df_troop.groupby('canonical_cookie_type'):
        # If not enough data, fallback to last known
        if group['period'].nunique() < 2:
            last_period = group['period'].max()
            last_val = group.loc[group['period'] == last_period, 'number_cases_sold'].mean()
            predicted_cases = last_val
        else:
            X_train = group[['period', 'period_squared', 'number_of_girls']]
            y_train = group['number_cases_sold']
            X_train = sm.add_constant(X_train)

            try:
                model = sm.OLS(y_train, X_train).fit()
                period_squared = chosen_period ** 2
                X_test = np.array([[1, chosen_period, period_squared, chosen_num_girls]])
                predicted_cases = model.predict(X_test)[0]
            except Exception as e:
                # If there's an error, skip
                continue

        # Clip predicted cases to historical min/max
        historical_low = group['historical_low'].iloc[0]
        historical_high = group['historical_high'].iloc[0]
        clipped_prediction = max(historical_low, min(predicted_cases, historical_high))

        interval_lower = max(clipped_prediction - interval_width, 1)
        interval_upper = clipped_prediction + interval_width

        predictions.append({
            "cookie_type": cookie_type,
            "predicted_cases": round(clipped_prediction, 2),
            "interval_lower": round(interval_lower, 2),
            "interval_upper": round(interval_upper, 2)
        })

    return jsonify(predictions)

# ----------------------------------------------------------------
# 6. RUN APP
# ----------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
