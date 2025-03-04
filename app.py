from flask import Flask, request, render_template, url_for
import pandas as pd
import statsmodels.api as sm
import numpy as np
import warnings
import time
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

warnings.simplefilter("ignore", category=RuntimeWarning)

app = Flask(__name__)

# Load dataset
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

# Define cookie images mapping
cookie_images = {
    "Adventurefuls": "adventurefuls.png",
    "Do-Si-Dos": "do-si-dos.png",
    "Lemon-Ups": "lemon-ups.png",
    "Samoas": "samoas.png",
    "Smores": "smores.png",
    "Tagalongs": "tagalongs.png",
    "Thin Mints": "thin-mints.png",
    "Toffee-tastic": "toffee-tastic.png",
    "Trefoils": "trefoils.png"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        troop_ids = sorted(df['troop_id'].unique().tolist())
        return render_template('index.html', troop_ids=troop_ids)

    chosen_year = request.form.get('year')
    chosen_troop = request.form.get('troop_id')
    chosen_num_girls = request.form.get('number_of_girls')

    try:
        chosen_period = int(chosen_year)
        chosen_troop = int(chosen_troop)
        chosen_num_girls = float(chosen_num_girls)
    except ValueError:
        return "Invalid input. Please enter valid numeric values.", 400

    if chosen_num_girls == 0:
        return (f"<h1>Predictions for Troop: {chosen_troop}, Period: {chosen_period}</h1>"
                f"<p>Number of Girls: {chosen_num_girls}</p>"
                f"<p>Since there are zero girls, no cookies will be sold.</p>")

    df_troop = df[(df['troop_id'] == chosen_troop) & (df['period'] < chosen_period)]
    if df_troop.empty:
        return f"No historical data found for troop {chosen_troop} before period {chosen_period}.", 404

    predictions = []
    for cookie_type, group in df_troop.groupby('cookie_type'):
        if group['period'].nunique() < 2:
            last_period = group['period'].max()
            last_val = group.loc[group['period'] == last_period, 'number_cases_sold'].mean()
            interval_lower = max(last_val - 10, 1)
            interval_upper = last_val + 10
            predictions.append({
                "cookie_type": cookie_type,
                "predicted_cases": round(last_val, 2),
                "interval_lower": round(interval_lower, 2),
                "interval_upper": round(interval_upper, 2),
                "image_path": url_for('static', filename=f'images/{cookie_images.get(cookie_type, "default.png")}')
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

            interval_lower = max(predicted_cases - 10, 1)
            interval_upper = predicted_cases + 10

            predictions.append({
                "cookie_type": cookie_type,
                "predicted_cases": round(predicted_cases, 2),
                "interval_lower": round(interval_lower, 2),
                "interval_upper": round(interval_upper, 2),
                "image_path": url_for('static', filename=f'images/{cookie_images.get(cookie_type, "default.png")}')
            })
        except:
            continue

    return render_template('predictions.html', predictions=predictions, troop_id=chosen_troop, period=chosen_period, num_girls=chosen_num_girls)

if __name__ == '__main__':
    app.run(debug=True)
