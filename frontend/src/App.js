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

// A helper to convert period 1->2020, 2->2021, etc.
function periodToYear(period) {
  // If period=1 => year=2020, so year = 2019 + period
  return 2019 + period;
}

function App() {
  const [troopId, setTroopId] = useState("");
  const [numGirls, setNumGirls] = useState("");
  const [predictions, setPredictions] = useState([]);

  // Autocomplete
  const [allTroopIds, setAllTroopIds] = useState([]);
  const [suggestions, setSuggestions] = useState([]);

  // Historical data
  const [pastSalesData, setPastSalesData] = useState([]);
  const [girlsData, setGirlsData] = useState([]);
  const [cookieBreakdownData, setCookieBreakdownData] = useState([]);
  const [cookieTypes, setCookieTypes] = useState([]);

  // Fetch troop IDs for autocomplete
  useEffect(() => {
    fetch("http://127.0.0.1:5000/api/troop_ids")
      .then((res) => res.json())
      .then((data) => setAllTroopIds(data))
      .catch((err) => console.error("Error fetching troop IDs:", err));
  }, []);

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

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!troopId || !numGirls) {
      alert("Please enter both Troop ID and Number of Girls.");
      return;
    }

    try {
      // 1) Predictions
      const resPred = await fetch("http://127.0.0.1:5000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          troop_id: troopId,
          num_girls: numGirls,
          year: 2024, // period=5 => 2024
        }),
      });
      const dataPred = await resPred.json();
      setPredictions(dataPred);

      // 2) Past sales & girls
      const resHist = await fetch(`http://127.0.0.1:5000/api/history/${troopId}`);
      const dataHist = await resHist.json();
      if (!dataHist.error) {
        setPastSalesData(dataHist.totalSalesByPeriod);
        setGirlsData(dataHist.girlsByPeriod);
      }

      // 3) Cookie breakdown for stacked bar
      const resBreak = await fetch(`http://127.0.0.1:5000/api/cookie_breakdown/${troopId}`);
      const dataBreak = await resBreak.json();
      setCookieBreakdownData(dataBreak);

      if (dataBreak.length > 0) {
        const keys = Object.keys(dataBreak[0]).filter((k) => k !== "period");
        setCookieTypes(keys);
      } else {
        setCookieTypes([]);
      }
    } catch (error) {
      console.error("Error fetching data:", error);
      alert("Error fetching predictions or historical data.");
    }
  };

  return (
    <div className="app-container">
      {/* NAVBAR with images in top-right */}
      <nav className="navbar">
        <div className="navbar-left">
          <h1>Cookie Sales Predictor</h1>
        </div>
        <div className="navbar-right">
          {/* If the images are in /public, just reference them as below */}
          <img
            src="http://127.0.0.1:5000/static/purdue.jpg"
            alt="Purdue"
            className="header-logo"
          />
          <img
            src="http://127.0.0.1:5000/static/gsci.jpg"
            alt="GSCI"
            className="header-logo"
          />
        </div>
      </nav>

      {/* Form */}
      <div className="form-container">
        <form onSubmit={handlePredict}>
          <label>Troop ID</label>
          <input
            type="text"
            value={troopId}
            onChange={handleTroopChange}
            placeholder="e.g. 123"
          />
          {suggestions.length > 0 && (
            <ul className="suggestions-list">
              {suggestions.map((id) => (
                <li key={id} onClick={() => handleSuggestionClick(id)}>
                  {id}
                </li>
              ))}
            </ul>
          )}

          <label>Number of Girls Participating in 2024</label>
          <input
            type="number"
            value={numGirls}
            onChange={(e) => setNumGirls(e.target.value)}
            placeholder="e.g. 20"
          />

          <button type="submit">Predict</button>
        </form>
      </div>

      {/* Predictions in a 3-wide grid */}
      {predictions.length > 0 && (
        <div className="predictions-container">
          <h2>Predictions</h2>
          <div className="predictions-grid">
            {predictions.map((p) => (
              <div key={p.cookie_type} className="prediction-card">
                <img
                  src={p.image_url}
                  alt={p.cookie_type}
                  style={{ width: 200, height: "auto" }}
                />
                <h3>{p.cookie_type}</h3>
                <p>Predicted Cases: {p.predicted_cases}</p>
                <p>
                  Interval: [{p.interval_lower}, {p.interval_upper}]
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Past Sales + Girls charts */}
      {pastSalesData.length > 0 && girlsData.length > 0 && (
        <div className="breakdown-section">
          <h2>Past Sales Breakdown (Troop {troopId})</h2>
          <div className="chart-row">
            <div className="chart-container">
              <h4>Total Cookie Sales by Year</h4>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={pastSalesData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="period"
                    tickFormatter={periodToYear} // Convert period to year
                  />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="totalSales"
                    stroke="#8884d8"
                    strokeWidth={3}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-container">
              <h4>Number of Girls (Avg) by Year</h4>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={girlsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="period"
                    tickFormatter={periodToYear} // Convert period to year
                  />
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

      {/* Stacked Bar: Cookie Breakdown */}
      {cookieBreakdownData.length > 0 && (
        <div className="breakdown-section">
          <h2>Cumulative Cookie Breakdown by Year</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={cookieBreakdownData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="period"
                  tickFormatter={periodToYear} // Convert period to year
                />
                <YAxis />
                <Tooltip />
                <Legend />
                {/* For each cookie type, add a stacked Bar */}
                {cookieTypes.map((ct, idx) => (
                  <Bar
                    key={ct}
                    dataKey={ct}
                    stackId="a"
                    fill={getColor(idx)}
                    name={ct}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

// Simple color function for stacked bars
function getColor(idx) {
  const palette = [
    "#8884d8", "#82ca9d", "#ffc658", "#d0ed57", "#a4de6c",
    "#8dd1e1", "#d88884", "#ad8de1", "#84d8a4", "#e1cf8d"
  ];
  return palette[idx % palette.length];
}

export default App;
