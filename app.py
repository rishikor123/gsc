from flask import Flask, request, jsonify
import pandas as pd
import statsmodels.api as sm
import numpy as np
import warnings

# Suppress warnings from statsmodels
warnings.simplefilter("ignore", category=RuntimeWarning)

app = Flask(__name__)

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
df = pd.read_csv('FinalCookieSales.csv')
df = df.drop(columns=['date'], errors='ignore')
df['number_cases_sold'] = pd.to_numeric(df['number_cases_sold'], errors='coerce')
df['period'] = pd.to_numeric(df['period'], errors='coerce')
df['number_of_girls'] = pd.to_numeric(df['number_of_girls'], errors='coerce')
df = df.dropna()
df = df[df['number_cases_sold'] > 0]
df['period_squared'] = df['period'] ** 2

# Calculate historical low & high for each troop-cookie type
historical_stats = df.groupby(['troop_id', 'cookie_type'])['number_cases_sold'].agg(['min', 'max']).reset_index()
historical_stats.columns = ['troop_id', 'cookie_type', 'historical_low', 'historical_high']

# Merge these guardrails into the dataset
df = df.merge(historical_stats, on=['troop_id', 'cookie_type'], how='left')

# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    troop_id = data.get('troop_id')
    
    try:
        number_of_girls = float(data.get('number_of_girls'))
    except Exception as e:
        return jsonify({"error": "Invalid number_of_girls input"}), 400

    # Filter dataset for the given troop_id
    df_troop = df[df['troop_id'] == troop_id]
    if df_troop.empty:
        return jsonify({"error": "Troop ID not found"}), 404

    predictions = []
    # Group by cookie_type for separate regressions
    for cookie_type, group in df_troop.groupby('cookie_type'):
        X = group[['period', 'period_squared', 'number_of_girls']]
        y = group['number_cases_sold']
        X = sm.add_constant(X)
        
        try:
            model = sm.OLS(y, X).fit()
            period = 5  # For 2024
            period_squared = period ** 2
            # Use the user input for number_of_girls for period 5 prediction
            pred_input = np.array([[1, period, period_squared, number_of_girls]])
            predicted_cases = model.predict(pred_input)[0]

            # Apply historical guardrails
            historical_low = group['historical_low'].iloc[0]
            historical_high = group['historical_high'].iloc[0]
            if predicted_cases < historical_low:
                predicted_cases = historical_low
            elif predicted_cases > historical_high:
                predicted_cases = historical_high

            predictions.append({
                "cookie_type": cookie_type,
                "predicted_cases": round(predicted_cases, 2)
            })
        except Exception as e:
            # In case the model fails for this cookie type, skip it.
            continue

    return jsonify({
        "troop_id": troop_id,
        "number_of_girls": number_of_girls,
        "predictions": predictions
    })

if __name__ == '__main__':
    app.run(debug=True)
