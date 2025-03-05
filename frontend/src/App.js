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

  // All troop IDs from the backend
  const [allTroopIds, setAllTroopIds] = useState([]);
  // Filtered suggestions to display under the input
  const [suggestions, setSuggestions] = useState([]);

  // Example data for Past Sales Breakdown (replace with real data if needed)
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

  // Fetch all troop IDs on mount
  useEffect(() => {
    fetch("/api/troop_ids")
      .then((res) => {
        if (!res.ok) {
          throw new Error("Failed to fetch troop IDs");
        }
        return res.json();
      })
      .then((data) => setAllTroopIds(data))
      .catch((err) => console.error("Error fetching troop IDs:", err));
  }, []);

  // Handle typing in the Troop ID input
  const handleTroopChange = (e) => {
    const value = e.target.value;
    setTroopId(value);

    if (!value) {
      setSuggestions([]);
      return;
    }
    // Filter allTroopIds to those that start with typed text
    const filtered = allTroopIds.filter((id) =>
      id.toString().startsWith(value)
    );
    setSuggestions(filtered);
  };

  // User clicks on a suggestion to fill the input
  const handleSuggestionClick = (id) => {
    setTroopId(id.toString());
    setSuggestions([]);
  };

  // Handle Predict button
  const handlePredict = async (e) => {
    e.preventDefault();
    if (!troopId || !numGirls) {
      alert("Please enter both a Troop ID and Number of Girls.");
      return;
    }

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          troop_id: troopId,
          num_girls: numGirls,
          year: 2024, // Hard-code to 2024 if desired
        }),
      });
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      const data = await response.json();
      setPredictions(data);
    } catch (error) {
      console.error("Error fetching predictions:", error);
      alert("There was an error fetching predictions.");
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

          {/* Autocomplete Suggestions */}
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

      {/* Only show predictions & charts if we have a non-empty response */}
      {predictions.length > 0 && (
        <div className="predictions-container">
          <h2>Predictions</h2>
          {predictions.map((p) => (
            <div key={p.cookie_type} className="prediction-card">
              <h3>{p.cookie_type}</h3>
              <p>Predicted Cases: {p.predicted_cases}</p>
              <p>Interval: [{p.interval_lower}, {p.interval_upper}]</p>
            </div>
          ))}

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
      )}
    </div>
  );
}

export default App;
