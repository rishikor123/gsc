// App.js
import React, { useState } from 'react';
import './manual.css';

function App() {
  const [isDetailsVisible, setIsDetailsVisible] = useState(false);
  const [lightMode, setLightMode] = useState(false);

  const handleShowDetails = () => {
    setIsDetailsVisible(true);
    setTimeout(() => {
      const section = document.getElementById('details');
      section && section.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  const toggleTheme = () => {
    setLightMode(!lightMode);
  };

  return (
    <div className={lightMode ? 'light-mode' : ''}>
      <div className="background"></div>
      <div className="overlay"></div>

      <div className="header">
        <img src="GSC(2).png" alt="GSCI Logo" />
        <img src="KREN.png" alt="KREN Logo" />
        <a href="MockGS.html" className="back-home">Back to Home</a>
      </div>

      <div className="title">Cookie Forecasting Manual</div>

      <div className="content">
        <p>Explore how predictive analytics can help Girl Scouts maximize cookie sales.</p>
        <p>Click the button below to explore data-driven insights, model findings, and outcomes.</p>
        <button className="view-details" onClick={handleShowDetails}>View Details</button>
        <button className="toggle-theme" onClick={toggleTheme}>Toggle Theme</button>
      </div>

      {isDetailsVisible && (
        <div className="details-section" id="details">
          <h2>Business Problem</h2>
          <p>Every year, thousands of Girl Scouts rely on cookie sales for fundraising. The current forecast method—based solely on the previous year’s numbers—explains only 70% of sales variability. This limitation often leads to missed revenue opportunities or excess stock. By leveraging advanced predictive models, we can bridge this gap and unlock new opportunities for growth, efficiency, and increased fundraising success.</p>

          <h2>Key Benefits</h2>
          <ul>
            <li>Improved inventory planning and cost savings</li>
            <li>Enhanced Marketing Strategies</li>
            <li>IncreasedTroop Engagement</li>
            <li>Data Driven Decision Making</li>
          </ul>

          <h2>Analytical Problem</h2>
          <p><strong>Analytical Context:</strong> The context involves analyzing historical sales data and external factors to improve forecasting accuracy by troop and cookie type.</p>
          <p><strong>Challenges:</strong> Challenges include high variance in sales across troops and regions, impact from external factors like weather and local events, and ensuring model reliability for troop leaders.</p>
          <p><strong>Solution Focus:</strong> The solution focuses on implementing machine learning models to enhance forecasting accuracy and optimize inventory management.</p>

          <h2>Research Questions</h2>
          <ul>
            <li>Can machine learning models effectively integrate historical sales data, troop participation rates to improve the accuracy of cookie sales forecasts beyond traditional methods?</li>
            <li>How can insights from predictive models be used to optimize inventory management and marketing strategies for Girl Scout troops, ensuring that each troop meets demand without costly surplus or shortages?</li>
          </ul>

          <h2>Data Dictionary</h2>
          <img src="ch5.png" alt="Data Results" />
          <img src="Ch1.png" alt="Data Chart" />

          <h2>Model Selection</h2>
          <strong>ASSUMPTIONS:</strong>
          <ul>
            <li>The model assumes past sales patterns are predictive of future sales performance.</li>
            <li>No major disruptions in cookie availability, troop operations, or supply chains are expected.</li>
          </ul>
          <strong>LIMITATION:</strong>
          <ul>
            <li>The model may underperform for troops with limited or erratic historical data.</li>
          </ul>
          <strong>SPLITTING:</strong>
          <ul>
            <li>The dataset is grouped by troop ID and cookie type. For each group, the data is split by year into Training (2020–2023) and Testing (2024).</li>
            <li>Cluster-based modeling is applied within each location and cookie type, allowing for more personalized predictions.</li>
          </ul>
          <strong>BASE MODEL APPROACHES:</strong>
          <ul>
            <li><strong>SIO Model:</strong> Uses last year’s sales and troop participation to estimate.</li>
            <li><strong>AVG Model:</strong> Averages past sales from 2021-2023 to predict 2024. However, these models had higher RMSE values, indicating significant prediction errors.</li>
          </ul>
          <img src="Ch2.jpg" alt="Base Model Chart" />
          <p>To improve accuracy, we built a Hybrid Multi-Model system that automatically selects the best among Clustered Ridge Regression, Troop-Level Ridge Regression, Linear Regression, SIO & Average, and Location-Level Ridge Regression. Each troop-cookie prediction uses the method with the lowest error, dynamically chosen based on past performance. We tested other models and found the Hybrid had the best performance.</p>
          <img src="ch4.jpg" alt="Model Selection Chart" />

          <h2>Validation</h2>
          <p><strong>Confidence in Predictions:</strong> Model predictions are validated using cross-validation (CV) within each training group to optimize regularization strength (λ), minimizing overfitting and ensuring stable performance across troops and cookie types.</p>
          <p><strong>Dynamic Error-Based Method Selection:</strong> For each troop-cookie pair, the model dynamically selects the prediction method with the lowest expected error (MSE), based on past performance. This approach ensures predictions are customized and evidence-driven.</p>
          <p><strong>Robustness Across Segments:</strong> The use of clustering + Ridge + fallback heuristics allows the model to adapt to sparse, dense, and even noisy troop histories—leading to consistent accuracy improvements over baseline models.</p>

          <h2>Final Model Results</h2>
          <img src="metric.jpg" alt="Metrics" />

          <h2>Key Findings</h2>
          <p>Our Hybrid Multi-Model approach, which blends Ridge Regression (with CV), linear models, and PGA heuristics, achieved the highest R² and lowest error metrics, highlighting its adaptability and accuracy. The model dynamically selects the best prediction method (from 6 options) for each troop-cookie pair, based on historical performance and data quality.</p>
          <p>By improving prediction accuracy by 1.35 cases per troop per cookie type compared to the SIO tool, our model provided a significant advantage in planning and inventory. Scaled across 1,401 troops, 8 cookie types, and 12 boxes per case, this translates to a potential impact of over 181,000 boxes—equivalent to more than</p>
          <img src="KF.JPG" alt="Key Findings Graphic" style={{ marginTop: '20px', borderRadius: '10px' }} />
          <p>This enables GS Indiana to optimize inventory, reduce over-ordering, and boost revenue for smarter, more profitable decisions.</p>

          <h2>Model Life Cycle</h2>
          <img src="ch6.jpg" alt="Model Life Cycle" />

          <div className="pdf-link">
            <a href="SC.pdf" target="_blank" rel="noopener noreferrer">View Full Poster PDF</a>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;



