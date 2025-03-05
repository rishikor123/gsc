import React, { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
} from "recharts";

function App() {
  const [troopId, setTroopId] = useState("");
  const [numGirls, setNumGirls] = useState("");
  const [predictions, setPredictions] = useState([]);

  const handlePredict = async (e) => {
    e.preventDefault();

    // Example: POST to your Flask endpoint /api/predict
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          troop_id: troopId,
          num_girls: numGirls,
          year: 2024,  // Hard-coded if you like
        }),
      });
      const data = await response.json();
      setPredictions(data);
    } catch (error) {
      console.error("Error fetching predictions:", error);
    }
  };

  // Example past sales data for charts
  const pastSalesData = [
    { year: 2020, totalSales: 220 },
    { year: 2021, totalSales: 280 },
    { year: 2022, totalSales: 340 },
    { year: 2023, totalSales: 390 },
  ];
  const girlsParticipationData = [
    { year: 2020, numberOfGirls: 15 },
    { year: 2021, numberOfGirls: 18 },
    { year: 2022, numberOfGirls: 20 },
    { year: 2023, numberOfGirls: 25 },
  ];

  return (
    <div className="app-container">
      <nav className="navbar">
        <h1>Cookie Sales Predictor</h1>
      </nav>

      <div className="form-container">
        <form onSubmit={handlePredict}>
          <label>Troop ID</label>
          <input
            type="text"
            value={troopId}
            onChange={(e) => setTroopId(e.target.value)}
            placeholder="123"
            required
          />

          <label>Number of Girls</label>
          <input
            type="number"
            value={numGirls}
            onChange={(e) => setNumGirls(e.target.value)}
            placeholder="20"
            required
          />

          <button type="submit">Predict</button>
        </form>
      </div>

      {predictions.length > 0 && (
        <div className="predictions-container">
          <h2>Predictions</h2>
          {predictions.map((p) => (
            <div key={p.cookie_type} className="prediction-card">
              <h3>{p.cookie_type}</h3>
              <p>Predicted Cases: {p.predicted_cases}</p>
              <p>
                Interval: [{p.interval_lower}, {p.interval_upper}]
              </p>
            </div>
          ))}
        </div>
      )}

      <div className="breakdown-section">
        <h2>Past Sales Breakdown</h2>
        <div className="chart-row">
          <div className="chart-container">
            <h4>Total Sales by Year</h4>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={pastSalesData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="totalSales"
                  stroke="#8884d8"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-container">
            <h4>Girls Participating</h4>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={girlsParticipationData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="numberOfGirls" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
