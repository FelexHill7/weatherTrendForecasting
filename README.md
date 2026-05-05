# Weather Trend Forecasting

## PM Accelerator Tech Assessment — Advanced Track

---

## PM Accelerator Mission

> PM Accelerator is committed to democratizing product management education globally.
> Their mission: **"Break down financial barriers and achieve educational fairness"** —
> offering free Product Management education to teenagers from underserved families through
> the **PMA KIDS** initiative, with a goal of establishing 200 schools worldwide over the
> next 20 years to foster diversity in tech and empower the next generation of product leaders.
>
> *"Join our global community of aspiring and current Product Managers and fast-track your career TODAY."*
> — [pmaccelerator.io](https://www.pmaccelerator.io)

---

## Project Overview

A comprehensive end-to-end data science project analyzing the **Global Weather Repository** dataset
to forecast weather trends. Implements both basic and advanced assessment requirements including
anomaly detection, multiple forecasting models, an ensemble approach, climate analysis, air quality
correlation, feature importance, and spatial visualization.

## Dataset

**Global Weather Repository** — available on Kaggle:
[https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)

- 40+ features per observation
- Worldwide daily/hourly weather data
- Covers temperature, precipitation, wind, air quality, astronomical data, and more

> **Note:** The notebook downloads the dataset automatically via `kagglehub` when Kaggle
> credentials are configured. A local CSV at `data/GlobalWeatherRepository.csv` is used
> as a fallback if kagglehub is unavailable.

## Repository Structure

```text
weatherTrendForecasting/
│   └── GlobalWeatherRepository.csv   
├── weather_forecasting.ipynb          ← main analysis notebook
├── requirements.txt
└── README.md
```

## Analysis Sections

| # | Section | Key Techniques |
| --- | --- | --- |
| 1 | Data Loading & Exploration | Shape, dtypes, descriptive stats |
| 2 | Data Cleaning & Preprocessing | Missing value imputation, Winsorization, feature engineering, normalization |
| 3 | Exploratory Data Analysis | Temperature/precipitation/wind/humidity distributions, correlations, geographic patterns |
| 4 | Anomaly Detection | Z-score, IQR (3×), Isolation Forest, Local Outlier Factor (LOF), PCA visualization |
| 5 | Time Series Analysis | ADF stationarity test, seasonal decomposition, ACF/PACF |
| 6 | Forecasting — SARIMA | SARIMA(1,0,1)(1,1,1,12) on monthly aggregation, 6-month forecast |
| 7 | Forecasting — Prophet | Trend + yearly/weekly seasonality components |
| 8 | Forecasting — XGBoost | Lag features, rolling statistics, date features |
| 9 | Ensemble Model | Inverse-RMSE weighted average of all models |
| 10 | Model Comparison | MAE, RMSE, R², MAPE table |
| 11 | Climate Analysis | Monthly normals by continent, variability heatmap, precipitation patterns |
| 12 | Air Quality & Environmental Impact | PM2.5/PM10 distributions, AQ vs weather correlation, WHO guidelines |
| 13 | Feature Importance | Random Forest importance, Mutual Information, SHAP values |
| 14 | Spatial Analysis | Global temperature/precipitation scatter maps, interactive Folium heatmap |

## Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd weatherTrendForecasting
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Prophet** requires additional system dependencies on some platforms.
> See the [Prophet installation guide](https://facebook.github.io/prophet/docs/installation.html)
> if the standard pip install fails.


### 5. Launch Jupyter and run the notebook

```bash
jupyter notebook weather_forecasting.ipynb
```

Or with JupyterLab:

```bash
jupyter lab weather_forecasting.ipynb
```

Run all cells: **Kernel → Restart & Run All**

## Key Libraries

| Library | Purpose |
| --- | --- |
| pandas, numpy | Data manipulation |
| matplotlib, seaborn | Static visualizations |
| folium | Interactive geographic maps |
| plotly | Additional interactive charts |
| scikit-learn | ML models, preprocessing, metrics |
| xgboost | Gradient-boosted forecasting |
| statsmodels | SARIMA time series model |
| prophet | Trend/seasonality decomposition and forecasting |
| shap | Model explainability (SHAP values) |
| scipy | Statistical tests |
| kagglehub | Automatic dataset download from Kaggle |

## Output Files

After running the notebook, images are saved to `plots/` and the interactive map to the project root:

| File | Description |
| --- | --- |
| `plots/missing_values.png` | Missing data bar chart |
| `plots/outlier_analysis.png` | Box plots + distributions for key features |
| `plots/temperature_analysis.png` | Global temperature EDA dashboard |
| `plots/precipitation_wind_humidity.png` | Precip, wind, humidity analysis |
| `plots/correlation_heatmap.png` | Feature correlation matrix |
| `plots/geographic_patterns.png` | Temperature by country and continent |
| `plots/weather_conditions.png` | Weather condition category breakdown |
| `plots/statistical_anomalies.png` | Z-score anomalies on time series |
| `plots/ml_anomaly_detection.png` | Isolation Forest & LOF (PCA projection) |
| `plots/time_series_decomposition.png` | Trend + seasonality + residual |
| `plots/acf_pacf.png` | Autocorrelation and partial autocorrelation |
| `plots/sarima_forecast.png` | SARIMA test forecast + 6-month projection |
| `plots/prophet_forecast.png` | Prophet forecast plot |
| `plots/prophet_components.png` | Prophet trend/seasonality components |
| `plots/xgboost_forecast.png` | XGBoost test forecast |
| `plots/model_comparison.png` | All models vs actual on test period |
| `plots/model_rmse_comparison.png` | Model RMSE bar chart |
| `plots/climate_normals.png` | Monthly climate normals by continent |
| `plots/climate_variability.png` | Temperature variability heatmap |
| `plots/precipitation_by_continent.png` | Monthly precipitation by continent |
| `plots/air_quality_distributions.png` | PM2.5, PM10, NO2, O3 distributions |
| `plots/aq_scatter.png` | Air quality scatter plots vs weather variables |
| `plots/aq_weather_correlation.png` | Air quality vs weather correlation heatmap |
| `plots/pm25_by_country.png` | Top polluted countries (PM2.5) |
| `plots/feature_importance.png` | RF importance + mutual information |
| `plots/shap_values.png` | SHAP summary plots |
| `plots/temperature_map.png` | Global temperature scatter map |
| `plots/precipitation_map.png` | Global precipitation scatter map |
| `interactive_weather_map.html` | Interactive Folium heatmap (root directory) |

## Methodology Summary

### Data Cleaning

- **Missing values**: median fill for numeric columns (<30% missing), mode fill for categorical; columns with >30% missing are dropped.
- **Outliers**: Winsorization (cap at 1.5×IQR bounds) preserves sample size while removing extremes.
- **Feature engineering**: datetime decomposition (year, month, day, hour, quarter, season), hemisphere-aware seasons, apparent temperature (Steadman formula).
- **Normalization**: StandardScaler applied to all numeric features for ML use.

### Time Series Construction

- Filtered to Western/Central Europe (35–70°N, 10°W–40°E) to obtain a coherent annual seasonal signal. Global NH averaging mixes incompatible climate zones (Arctic Norway, tropical India) that blur the seasonal wave SARIMA needs.
- Resampled to daily means; short gaps filled with linear time interpolation (max 3 days).
- Rolling median outlier correction (14-day window, 3σ threshold with 1.5°C minimum floor) removes data-quality spikes without over-smoothing seasonal transitions.
- Train/test split snapped to a calendar month boundary (~85%/15%) so SARIMA trains on complete seasonal cycles.

### EDA

- Distribution plots, box plots, violin plots, scatter plots across temperature, precipitation, wind, humidity, and UV index.
- Correlation analysis via Pearson matrix and Mutual Information.
- Geographic and seasonal patterns explored at city, country, and continental level.

### Anomaly Detection

- **Statistical**: Z-score (|z| > 3) and IQR (3× fence) — fast, interpretable.
- **ML-based**: Isolation Forest and LOF (contamination = 3%) — captures multivariate anomalies.
- Consensus anomalies (flagged by both methods) are profiled and visualized.

### Forecasting Models

1. **SARIMA(1,0,1)(1,1,1,12)** — seasonal ARIMA on monthly-aggregated European temperature data. Period=12 captures the annual seasonal cycle. Monthly aggregation is used because daily SARIMA with period=365 is computationally prohibitive and the dataset spans ~2 years (insufficient for daily seasonal estimation).
2. **Prophet** — Facebook's trend + seasonality decomposition model; handles missing data and multiple seasonalities robustly.
3. **XGBoost** — gradient-boosted trees on lag features (1, 2, 3, 7, 14, 21, 30 days), rolling statistics (7/14/30-day windows), and date features (month, hour, day-of-week, quarter).
4. **Ensemble** — inverse-RMSE weighted average of all three models; SARIMA contributes monthly-interpolated daily values.

Models evaluated on a season-aligned held-out test set (split at a calendar month boundary, ~15% of data) using MAE, RMSE, R², and MAPE.

### Advanced Analyses

- **Climate analysis**: monthly normals and variability by continent, precipitation seasonality.
- **Environmental impact**: PM2.5/PM10 concentration benchmarked against WHO and US EPA standards; correlation with wind speed, temperature, and humidity.
- **Feature importance**: Random Forest importance scores, Mutual Information, and SHAP tree explainability.
- **Spatial analysis**: latitude/longitude scatter maps colored by temperature and precipitation; interactive Folium heatmap with city-level tooltips.

## Results Summary

| Model | Evaluation |
| --- | --- |
| SARIMA | Monthly seasonal baseline; captures annual temperature cycle via period=12 |
| Prophet | Robust trend detection; good for long-range projections |
| XGBoost | Best short-term accuracy; highly sensitive to lag features |
| **Ensemble** | **Lowest RMSE overall**; benefits from model diversity |

Key findings:

- Feels-like temperature (`feels_like_celsius`) is the single strongest predictor of ambient temperature.
- Wind speed is the strongest negative correlate of PM2.5 air pollution.
- Continental interiors have the highest seasonal temperature variability.
- The ensemble model consistently outperforms any individual model on the test set.
