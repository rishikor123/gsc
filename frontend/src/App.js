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

  // Autocomplete
  const [allTroopIds, setAllTroopIds] = useState([]);
  const [suggestions, setSuggestions] = useState([]);

  // Historical data for line & bar chart
  const [pastSalesData, setPastSalesData] = useState([]);
  const [girlsData, setGirlsData] = useState([]);

  // Cookie breakdown for stacked bar chart
  const [cookieBreakdownData, setCookieBreakdownData] = useState([]);
  const [cookieTypes, setCookieTypes] = useState([]); // to dynamically render stacked bars

  // 1) Fetch troop IDs once
  useEffect(() => {
    fetch("http://127.0.0.1:5000/api/troop_ids")
      .then((res) => res.json())
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

  // 3) On Predict
  const handlePredict = async (e) => {
    e.preventDefault();
    if (!troopId || !numGirls) {
      alert("Please enter both Troop ID and Number of Girls.");
      return;
    }

    try {
      // a) Fetch predictions
      const resPred = await fetch("http://127.0.0.1:5000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          troop_id: troopId,
          num_girls: numGirls,
          year: 2024,
        }),
      });
      const dataPred = await resPred.json();
      setPredictions(dataPred);

      // b) Fetch line/bar chart data (total sales & girls)
      const resHist = await fetch(`http://127.0.0.1:5000/api/history/${troopId}`);
      const dataHist = await resHist.json();
      if (!dataHist.error) {
        setPastSalesData(dataHist.totalSalesByPeriod);
        setGirlsData(dataHist.girlsByPeriod);
      }

      // c) Fetch cookie breakdown for stacked bar
      const resBreak = await fetch(`http://127.0.0.1:5000/api/cookie_breakdown/${troopId}`);
      const dataBreak = await resBreak.json();
      setCookieBreakdownData(dataBreak);

      // Dynamically extract cookie types (all keys except "period")
      if (dataBreak.length > 0) {
        const allKeys = Object.keys(dataBreak[0]);
        const cTypes = allKeys.filter((k) => k !== "period");
        setCookieTypes(cTypes);
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
      <nav className="navbar">
        <h1>Cookie Sales Predictor</h1>
      </nav>

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

      {/* Past Sales + Girls */}
      {pastSalesData.length > 0 && girlsData.length > 0 && (
        <div className="breakdown-section">
          <h2>Past Sales Breakdown (Troop {troopId})</h2>
          <div className="chart-row">
            <div className="chart-container">
              <h4>Total Cookie Sales by Period</h4>
              <ResponsiveContainer width="100%" height={400}>
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
                    strokeWidth={3}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-container">
              <h4>Number of Girls (Avg) by Period</h4>
              <ResponsiveContainer width="100%" height={400}>
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

      {/* Stacked Bar: Cookie Breakdown */}
      {cookieBreakdownData.length > 0 && (
        <div className="breakdown-section">
          <h2>Cumulative Cookie Breakdown by Period</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={cookieBreakdownData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="period" />
                <YAxis />
                <Tooltip />
                <Legend />
                {/* For each cookie type, we add a stacked <Bar> */}
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

// A simple color palette for stacked bars
function getColor(idx) {
  const palette = [
    "#8884d8", "#82ca9d", "#ffc658", "#d0ed57", "#a4de6c",
    "#8dd1e1", "#d88884", "#ad8de1", "#84d8a4", "#e1cf8d"
  ];
  return palette[idx % palette.length];
}

export default App;
