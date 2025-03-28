// App.js
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
import "./index.css";

const cookies = [
  { name: "Adventurefuls", image: "ADVEN.png" },
  { name: "Do-Si-Dos", image: "DOSI.png" },
  { name: "Lemon-Ups", image: "LMNUP.png" },
  { name: "Samoas", image: "SAM.png" },
  { name: "Tagalongs", image: "TAG.png" },
  { name: "Thin Mints", image: "THIN.png" },
  { name: "Toffee-Tastic", image: "TFTAS.png" },
  { name: "Trefoils", image: "TREF.png" },
  { name: "S'mores", image: "SMORE.png" },
];

function periodToYear(period) {
  return 2019 + period;
}

function App() {
  const [troopId, setTroopId] = useState("");
  const [numGirls, setNumGirls] = useState("");
  const [suUnit, setSuUnit] = useState("");
  const [predictions, setPredictions] = useState({});
  const [pastSalesData, setPastSalesData] = useState([]);
  const [girlsData, setGirlsData] = useState([]);
  const [cookieBreakdownData, setCookieBreakdownData] = useState([]);
  const [cookieTypes, setCookieTypes] = useState([]);

  const handlePredict = async () => {
    try {
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
      const formatted = {};
      dataPred.forEach((d) => {
        formatted[d.cookie_type] = {
          predictedCases: d.predicted_cases,
          interval: [d.interval_lower, d.interval_upper],
        };
      });
      setPredictions(formatted);

      const resHist = await fetch(`http://127.0.0.1:5000/api/history/${troopId}`);
      const dataHist = await resHist.json();
      if (!dataHist.error) {
        setPastSalesData(dataHist.totalSalesByPeriod);
        setGirlsData(dataHist.girlsByPeriod);
      }

      const resBreak = await fetch(`http://127.0.0.1:5000/api/cookie_breakdown/${troopId}`);
      const dataBreak = await resBreak.json();
      setCookieBreakdownData(dataBreak);
      if (dataBreak.length > 0) {
        const keys = Object.keys(dataBreak[0]).filter((k) => k !== "period");
        setCookieTypes(keys);
      } else {
        setCookieTypes([]);
      }
    } catch (err) {
      console.error("Error fetching prediction or analytics:", err);
    }
  };

  return (
    <div className="main-container">
      <div className="background"></div>
      <div className="overlay"></div>

      <header className="header">
        <div>
          <img src="GSC(2).png" alt="GSCI Logo" />
          <img src="KREN.png" alt="KREN Logo" />
        </div>
        <a href="manual.html" className="manual">Manual</a>
      </header>

      <h1 className="title">Cookie Forecasting Model</h1>
      <p className="subtitle">Forecasting Sales, One Cookie at a Time</p>

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

      <div className="predictions">PREDICTIONS</div>
      <div className="cookie-grid">
        {cookies.map((cookie) => (
          <div key={cookie.name} className="cookie-box">
            <img src={cookie.image} alt={cookie.name} />
            <div className="cookie-info">
              <strong>{cookie.name}</strong>
              <br />
              Predicted Cases: <span>{predictions[cookie.name]?.predictedCases ?? "--"}</span>
              <br />
              Interval: <span>{predictions[cookie.name] ? `[${predictions[cookie.name].interval[0]}, ${predictions[cookie.name].interval[1]}]` : "--"}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="analytics-title">ANALYTICS</div>
      <div className="analysis-section">
        <div className="analysis-box">
          <h4>Total Cookie Sales by Year</h4>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={pastSalesData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="period" tickFormatter={periodToYear} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="totalSales" stroke="#8884d8" strokeWidth={3} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="analysis-box">
          <h4>Number of Girls (Avg) by Year</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={girlsData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="period" tickFormatter={periodToYear} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="numberOfGirls" fill="#82ca9d" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="analysis-box" style={{ gridColumn: "span 2" }}>
          <h4>Cumulative Cookie Breakdown by Year</h4>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={cookieBreakdownData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="period" tickFormatter={periodToYear} />
              <YAxis />
              <Tooltip />
              <Legend />
              {cookieTypes.map((ct, idx) => (
                <Bar key={ct} dataKey={ct} stackId="a" fill={getColor(idx)} name={ct} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function getColor(idx) {
  const palette = [
    "#8884d8",
    "#82ca9d",
    "#ffc658",
    "#d0ed57",
    "#a4de6c",
    "#8dd1e1",
    "#d88884",
    "#ad8de1",
    "#84d8a4",
    "#e1cf8d",
  ];
  return palette[idx % palette.length];
}

export default App;
