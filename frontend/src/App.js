import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts";
import "./index.css";

const periodToYear = (period) => 2019 + period;

const App = () => {
  const [troopId, setTroopId] = useState("");
  const [numGirls, setNumGirls] = useState("");
  const [suUnit, setSuUnit] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [analytics, setAnalytics] = useState({ sales: [], girls: [], breakdown: [] });

  const handlePredict = async () => {
    if (!troopId || !numGirls) {
      alert("Please enter Troop ID and Number of Girls.");
      return;
    }

    try {
      const resPred = await fetch("http://localhost:5000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          troop_id: troopId,
          num_girls: numGirls,
          year: 2024
        })
      });
      const predData = await resPred.json();
      setPredictions(predData);

      const resHist = await fetch(`http://localhost:5000/api/history/${troopId}`);
      const histData = await resHist.json();

      const resBreak = await fetch(`http://localhost:5000/api/cookie_breakdown/${troopId}`);
      const breakData = await resBreak.json();

      setAnalytics({
        sales: histData.totalSalesByPeriod || [],
        girls: histData.girlsByPeriod || [],
        breakdown: breakData || []
      });
    } catch (err) {
      console.error("Error fetching data:", err);
    }
  };

  const getColor = (idx) => {
    const palette = [
      "#8884d8", "#82ca9d", "#ffc658", "#d0ed57", "#a4de6c",
      "#8dd1e1", "#d88884", "#ad8de1", "#84d8a4", "#e1cf8d"
    ];
    return palette[idx % palette.length];
  };

  const cookieKeys = analytics.breakdown.length > 0
    ? Object.keys(analytics.breakdown[0]).filter(key => key !== "period")
    : [];

  return (
    <div>
      <div className="background"></div>
      <div className="overlay"></div>
      <div className="header">
        <div>
          <img src="/static/GSC(2).png" alt="GSCI Logo" />
          <img src="/static/KREN.png" alt="KREN Logo" />
        </div>
        <a href="manual.html" className="manual">Manual</a>
      </div>

      <div className="title">Cookie Forecasting Model</div>
      <div className="subtitle">Forecasting Sales, One Cookie at a Time</div>

      <div className="input-container">
        <p>Enter the details below to forecast cookie sales</p>
        <div className="input-box">
          Enter Troop ID: <input type="text" value={troopId} onChange={(e) => setTroopId(e.target.value)} />
        </div>
        <div className="input-box">
          Enter Number of Girls Participating: <input type="text" value={numGirls} onChange={(e) => setNumGirls(e.target.value)} />
        </div>
        <div className="input-box">
          Enter SU Unit: <input type="text" value={suUnit} onChange={(e) => setSuUnit(e.target.value)} />
        </div>
        <button className="predict-button" onClick={handlePredict}>Predict</button>
      </div>

      {predictions.length > 0 && (
        <>
          <div className="predictions">PREDICTIONS</div>
          <div className="cookie-grid">
            {predictions.map((cookie, i) => (
              <div className="cookie-box" key={i}>
                <img src={cookie.image_url} alt={cookie.cookie_type} />
                <div className="cookie-info">
                  <strong>{cookie.cookie_type}</strong><br />
                  Predicted Cases: <span>{cookie.predicted_cases}</span><br />
                  Interval: <span>[{cookie.interval_lower}, {cookie.interval_upper}]</span>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {(analytics.sales.length > 0 || analytics.girls.length > 0 || analytics.breakdown.length > 0) && (
        <div className="analytics-title">ANALYTICS</div>
      )}
      <div className="analysis-section">
        <div className="analysis-box">
          <h4>Total Sales by Year</h4>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={analytics.sales}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="period" tickFormatter={periodToYear} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="totalSales" stroke="#8884d8" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="analysis-box">
          <h4>Number of Girls by Year</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={analytics.girls}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="period" tickFormatter={periodToYear} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="numberOfGirls" fill="#82ca9d" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="analysis-box">
          <h4>Cookie Breakdown by Year</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={analytics.breakdown}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="period" tickFormatter={periodToYear} />
              <YAxis />
              <Tooltip />
              <Legend />
              {cookieKeys.map((key, idx) => (
                <Bar key={key} dataKey={key} stackId="a" fill={getColor(idx)} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="analysis-box">
          <h4>More Analytics Coming Soon</h4>
        </div>
      </div>
    </div>
  );
};

export default App;
