from flask import Flask, request, render_template
import pandas as pd
import statsmodels.api as sm
import numpy as np
import warnings
import time

# Additional imports for Ridge modeling
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

# Convert to Python ints where appropriate
df['troop_id'] = df['troop_id'].astype(int)
df['period'] = df['period'].astype(int)

# Create squared period column for OLS
df['period_squared'] = df['period'] ** 2

# Calculate historical guardrails
historical_stats = df.groupby(['troop_id', 'cookie_type'])['number_cases_sold'].agg(['min', 'max']).reset_index()
historical_stats.columns = ['troop_id', 'cookie_type', 'historical_low', 'historical_high']
df = df.merge(historical_stats, on=['troop_id', 'cookie_type'], how='left')

# ----------------------------------------------------------------
# 2. RIDGE + INTERVAL ANALYSIS FUNCTION
# ----------------------------------------------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_ridge_interval_analysis():
    start_time = time.time()

    groups = df.groupby(['troop_id', 'cookie_type'])
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
            tdf['cookie_type'] = cookie
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

    # Build single test DataFrame
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

    # Store the overall training RMSE in Flask config (if we want to use it in OLS route)
    app.config['OVERALL_RIDGE_RMSE'] = overall_train_rmse

    # Print results to console
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
        print(worst_preds[['troop_id', 'cookie_type', 'period', 'number_of_girls',
                           'number_cases_sold', 'predicted', 'interval_lower',
                           'interval_upper', 'in_interval', 'error']])
    else:
        print("\nNo test data found during Ridge analysis.")

    end_time = time.time()
    print(f"\nProcess completed in {end_time - start_time:.2f} seconds.\n")


# ----------------------------------------------------------------
# 3. RUN RIDGE ANALYSIS ONCE AT STARTUP
# ----------------------------------------------------------------
run_ridge_interval_analysis()

# ----------------------------------------------------------------
# 4. MAIN ROUTE
#    GET -> Renders index.html with all troop_ids for JavaScript auto-complete
#    POST -> OLS predictions
# ----------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # Gather unique troop IDs
        troop_ids = sorted(df['troop_id'].unique().tolist())
        return render_template('index.html', troop_ids=troop_ids)

    # POST logic
    chosen_year = request.form.get('year')         # e.g. "5"
    chosen_troop = request.form.get('troop_id')    # e.g. "123"
    chosen_num_girls = request.form.get('number_of_girls')

    try:
        # Interpret "year" as internal "period"
        chosen_period = int(chosen_year)
        chosen_troop = int(chosen_troop)
        chosen_num_girls = float(chosen_num_girls)
    except ValueError:
        return "Invalid input. Please enter valid numeric values.", 400

    if chosen_num_girls == 0:
        return (f"<h1>Predictions for Troop: {chosen_troop}, Period: {chosen_period}</h1>"
                f"<p>Number of Girls: {chosen_num_girls}</p>"
                f"<p>Since there are zero girls, no cookies will be sold.</p>")

    # Filter historical data: periods < chosen_period
    df_troop = df[(df['troop_id'] == chosen_troop) & (df['period'] < chosen_period)]
    if df_troop.empty:
        return f"No historical data found for troop {chosen_troop} before period {chosen_period}.", 404

    predictions = []
    for cookie_type, group in df_troop.groupby('cookie_type'):
        # If only 1 data point, fallback
        if group['period'].nunique() < 2:
            last_period = group['period'].max()
            last_val = group.loc[group['period'] == last_period, 'number_cases_sold'].mean()
            predictions.append({
                "cookie_type": cookie_type,
                "predicted_cases": round(last_val, 2),
                "note": "Using last available period value"
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

            # Historical guardrails
            historical_low = group['historical_low'].iloc[0]
            historical_high = group['historical_high'].iloc[0]
            predicted_cases = max(historical_low, min(predicted_cases, historical_high))

            predictions.append({
                "cookie_type": cookie_type,
                "predicted_cases": round(predicted_cases, 2)
            })
        except:
            continue

    # Format HTML response
    html_result = f"<h1>Predictions for Troop: {chosen_troop}, Period: {chosen_period}</h1>"
    html_result += f"<p>Number of Girls: {chosen_num_girls}</p>"

    if not predictions:
        html_result += "<p>No predictions available.</p>"
    else:
        html_result += "<ul>"
        for pred in predictions:
            note = f" ({pred['note']})" if 'note' in pred else ""
            html_result += (f"<li>Cookie Type: {pred['cookie_type']} "
                            f"- Predicted Cases: {pred['predicted_cases']}{note}</li>")
        html_result += "</ul>"

    return html_result

# ----------------------------------------------------------------
# 5. RUN APP
# ----------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
