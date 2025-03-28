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
  const [predictions, setPredictions] = useState([]);
  const [pastSalesData, setPastSalesData] = useState([]);
  const [girlsData, setGirlsData] = useState([]);
  const [cookieBreakdownData, setCookieBreakdownData] = useState([]);
  const [cookieTypes, setCookieTypes] = useState([]);

  const predictSales = async () => {
    if (!troopId || !numGirls) {
      alert("Please enter Troop ID and Number of Girls.");
      return;
    }

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
      setPredictions(dataPred);

      const resHist = await fetch(`http://127.0.0.1:5000/api/history/${troopId}`);
      const dataHist = await resHist.json();
      if (!dataHist.error) {
        setPastSalesData(dataHist.totalSalesByPeriod);
        setGirlsData(dataHist.girlsByPeriod);
      }

      const resBreak = await fetch(`http://127.0.0.1:5000/api/cookie_breakdown/${troopId}`);
      const dataBreak = await resBreak.json();
      setCookieBreakdownData(dataBreak);
      const keys = dataBreak.length > 0 ? Object.keys(dataBreak[0]).filter((k) => k !== "period") : [];
      setCookieTypes(keys);
    } catch (err) {
      console.error("Error during prediction:", err);
    }
  };

  return (
    <div>
      <div className="background"></div>
      <div className="overlay"></div>

      <div className="header">
        <div>
          <img src="GSC(2).png" alt="GSCI Logo" />
          <img src="KREN.png" alt="KREN Logo" />
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
        <button className="predict-button" onClick={predictSales}>Predict</button>
      </div>

      {predictions.length > 0 && (
        <>
          <div className="predictions">PREDICTIONS</div>
          <div className="cookie-grid">
            {predictions.map((cookie) => (
              <div className="cookie-box" key={cookie.cookie_type}>
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

      {pastSalesData.length > 0 && girlsData.length > 0 && (
        <>
          <div className="analytics-title">ANALYTICS</div>
          <div className="analysis-section">
            <div className="chart-container">
              <h4>Total Cookie Sales by Year</h4>
              <ResponsiveContainer width="100%" height={400}>
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

            <div className="chart-container">
              <h4>Number of Girls (Avg) by Year</h4>
              <ResponsiveContainer width="100%" height={400}>
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
          </div>
        </>
      )}

      {cookieBreakdownData.length > 0 && (
        <div className="breakdown-section">
          <h2>Cumulative Cookie Breakdown by Year</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={cookieBreakdownData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="period" tickFormatter={periodToYear} />
                <YAxis />
                <Tooltip />
                <Legend />
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

function getColor(idx) {
  const palette = [
    "#8884d8", "#82ca9d", "#ffc658", "#d0ed57", "#a4de6c",
    "#8dd1e1", "#d88884", "#ad8de1", "#84d8a4", "#e1cf8d"
  ];
  return palette[idx % palette.length];
}

export default App;
