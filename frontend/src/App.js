import React, { useState, useEffect } from "react";
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

  // For autocomplete
  const [allTroopIds, setAllTroopIds] = useState([]);
  const [suggestions, setSuggestions] = useState([]);

  // Real historical data (from /api/history)
  const [pastSalesData, setPastSalesData] = useState([]);
  const [girlsData, setGirlsData] = useState([]);

  // 1) Load troop IDs for autocomplete
  useEffect(() => {
    fetch("http://127.0.0.1:5000/api/troop_ids")
      .then((res) => {
        if (!res.ok) {
          throw new Error("Failed to fetch troop IDs");
        }
        return res.json();
      })
      .then((data) => setAllTroopIds(data))
      .catch((err) => console.error("Error fetching troop IDs:", err));
  }, []);

  // 2) Autocomplete logic
  const handleTroopChange = (e) => {
    const value = e.target.value;
    setTroopId(value);

    if (!value) {
      setSuggestions([]);
      return;
    }
    const filtered = allTroopIds.filter((id) =>
      id.toString().startsWith(value)
    );
    setSuggestions(filtered);
  };

  const handleSuggestionClick = (id) => {
    setTroopId(id.toString());
    setSuggestions([]);
  };

  // 3) On Predict: fetch predictions, then fetch historical data
  const handlePredict = async (e) => {
    e.preventDefault();
    if (!troopId || !numGirls) {
      alert("Please enter both a Troop ID and Number of Girls.");
      return;
    }

    try {
      // 3a) POST to /api/predict
      const response = await fetch("http://127.0.0.1:5000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          troop_id: troopId,
          num_girls: numGirls,
          year: 2024, // e.g. period=5
        }),
      });
      if (!response.ok) {
        throw new Error(`Predict error: ${response.status}`);
      }
      const data = await response.json();
      setPredictions(data);

      // 3b) GET /api/history/<troopId>
      const histRes = await fetch(`http://127.0.0.1:5000/api/history/${troopId}`);
      if (!histRes.ok) {
        throw new Error(`History error: ${histRes.status}`);
      }
      const histData = await histRes.json();
      if (!histData.error) {
        setPastSalesData(histData.totalSalesByPeriod);  // array of {period, totalSales}
        setGirlsData(histData.girlsByPeriod);           // array of {period, numberOfGirls}
      }
    } catch (error) {
      console.error("Error fetching data:", error);
      alert("There was an error fetching predictions or history data.");
    }
  };

  return (
    <div className="app-container">
      <nav className="navbar">
        <h1>Cookie Sales Predictor</h1>
      </nav>

      {/* Form */}
      <div className="form-container">
        <form onSubmit={handlePredict}>
          <label htmlFor="troopId">Troop ID</label>
          <input
            id="troopId"
            type="text"
            value={troopId}
            onChange={handleTroopChange}
            placeholder="e.g. 123"
          />

          {/* Autocomplete suggestions */}
          {suggestions.length > 0 && (
            <ul className="suggestions-list">
              {suggestions.map((id) => (
                <li key={id} onClick={() => handleSuggestionClick(id)}>
                  {id}
                </li>
              ))}
            </ul>
          )}

          <label htmlFor="numGirls">Number of Girls Participating in 2024</label>
          <input
            id="numGirls"
            type="number"
            value={numGirls}
            onChange={(e) => setNumGirls(e.target.value)}
            placeholder="e.g. 20"
          />

          <button type="submit">Predict</button>
        </form>
      </div>

      {/* Predictions */}
      {predictions.length > 0 && (
        <div className="predictions-container">
          <h2>Predictions</h2>
          {predictions.map((p) => (
            <div key={p.cookie_type} className="prediction-card">
              <img
                src={p.image_url}
                alt={p.cookie_type}
                style={{ width: 120, height: "auto" }}
              />
              <h3>{p.cookie_type}</h3>
              <p>Predicted Cases: {p.predicted_cases}</p>
              <p>
                Interval: [{p.interval_lower}, {p.interval_upper}]
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Historical Charts */}
      {/* Only show if we actually have data (pastSalesData, girlsData) */}
      {pastSalesData.length > 0 && girlsData.length > 0 && (
        <div className="breakdown-section">
          <h2>Past Sales Breakdown (Troop {troopId})</h2>
          <div className="chart-row">
            <div className="chart-container">
              <h4>Total Cookie Sales by Period</h4>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={pastSalesData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="period" />
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
              <h4>Number of Girls (Avg) by Period</h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={girlsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="period" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="numberOfGirls" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
