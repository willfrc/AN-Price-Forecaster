# UK Ammonium Nitrate Price Forecaster

A machine learning pipeline for forecasting UK ammonium nitrate (AN) bulk spot prices. Produces weekly forecasts for the next 3 months and monthly directional forecasts out to 12 months, with confidence intervals.

Built as a proof of concept using publicly available free data sources.

---

## Project Structure

```
an_price_forecaster/
├── ingestion/
│   ├── fred_ingestion.py          # FRED API: TTF gas, Brent, USD/GBP FX
│   ├── yfinance_ingestion.py      # CF Industries, Yara, SOIL ETF equities
│   ├── gie_ingestion.py           # EU natural gas storage (GIE AGSI+)
│   ├── worldbank_ingestion.py     # World Bank Pink Sheet: urea, DAP, phosphate
│   ├── ahdb_ingestion.py          # AHDB UK AN spot price (manual CSV)
│   └── ember_ingestion.py         # EU ETS carbon price (manual CSV)
├── processing/
│   ├── build_features.py          # Aligns all sources to monthly, engineers features
│   └── prepare_model_data.py      # Train/test splits, feature selection by tier
├── model/
│   ├── train.py                   # XGBoost training for 1-month and 12-month horizons
│   ├── forecast.py                # Generates 12-month forecast with confidence intervals
│   ├── evaluate.py                # Test set evaluation and walk-forward backtest
│   ├── confidence_intervals.py    # Residual-based CI estimation
│   └── saved/                     # Trained model files (gitignored)
├── visualisation/
│   └── dashboard.py               # Generates self-contained HTML dashboard
├── utils/
│   ├── db.py                      # SQLite database manager
│   ├── logger.py                  # Centralised logging
│   └── config.py                  # Series IDs, tickers, file paths
├── notebooks/
│   └── 01_data_exploration.ipynb  # EDA notebook for VS Code
├── data/
│   ├── raw/                       # Manual CSV downloads (gitignored)
│   └── processed/                 # Feature CSVs and forecast outputs (gitignored)
├── run_ingestion.py               # Master ingestion runner
├── run_pipeline.py                # Full end-to-end pipeline runner
└── requirements.txt
```

---

## Setup

### 1. Prerequisites
- Python 3.10+
- VS Code (recommended)
- Git

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API keys
```bash
copy .env.example .env
```

Edit `.env` and add your keys:
```
FRED_API_KEY=your_key_here       # Free: https://fredaccount.stlouisfed.org/apikeys
GIE_API_KEY=your_key_here        # Free: https://agsi.gie.eu
```

### 5. Manual data downloads
Two sources require manual download (no public API):

| File | Source | Save as |
|------|--------|---------|
| AHDB UK AN prices | [ahdb.org.uk/dairy/uk-fertiliser-price-tracker](https://ahdb.org.uk/dairy/uk-fertiliser-price-tracker) | `data/raw/ahdb_an_prices.csv` |
| World Bank Pink Sheet | [worldbank.org/en/research/commodity-markets](https://www.worldbank.org/en/research/commodity-markets) | `data/raw/worldbank_pink_sheet.xlsx` |

Also create `data/raw/` if it doesn't exist:
```bash
mkdir data\raw
```

---

## Running the pipeline

### Full pipeline (recommended)
```bash
py run_pipeline.py --skip ingest    # Skip ingestion if data is already collected
py run_pipeline.py                  # Full run including data refresh
```

### Step by step
```bash
py run_ingestion.py                 # Collect all data sources
py processing/build_features.py     # Engineer features, align to monthly
py processing/prepare_model_data.py # Build train/test splits
py model/train.py                   # Train XGBoost models
py model/evaluate.py                # Evaluate on test set + walk-forward backtest
py model/forecast.py                # Generate 12-month forecast
py visualisation/dashboard.py       # Build HTML dashboard
```

### Selective ingestion
```bash
py run_ingestion.py --source fred yfinance    # Run specific sources only
py run_ingestion.py --skip ahdb ember         # Skip manual sources
```

---

## Data Sources

| Source | Data | Frequency | Notes |
|--------|------|-----------|-------|
| AHDB | UK AN bulk spot price (target variable) | Monthly | Manual download |
| FRED | TTF gas, Brent crude, USD/GBP FX | Monthly/Daily | Free API key |
| World Bank Pink Sheet | Urea, DAP, phosphate rock, European gas | Monthly | Manual download |
| yfinance | CF Industries, Yara, SOIL ETF | Daily | No key needed |
| GIE AGSI+ | EU natural gas storage levels | Daily | Free API key |
| Ember Climate | EU ETS carbon price | Weekly | Manual download (optional) |

---

## Model

### Architecture
Two XGBoost gradient boosting models trained independently:

**1-month horizon model**
- Target: UK AN price 1 month ahead
- Training data: ~70 monthly observations (Apr 2019 – Jun 2025)
- Test set: 12 months (Jul 2025 – Jun 2026)
- MAPE: ~7% | MAE: ~£32/tonne | Walk-forward MAPE: ~5.6%

**12-month horizon model**
- Target: UK AN price 12 months ahead
- Training data: ~59 monthly observations
- Test set: 12 months
- MAPE: ~11.6% | MAE: ~£52/tonne
- Treat as directional guidance — not a precise price target

### Top features (1-month model)
1. TTF gas price GBP (lagged 1 month) — ~48% importance
2. Global urea price GBP (lagged 1 month) — ~17%
3. DAP benchmark price — ~13%
4. Lagged AN price (1-3 months)
5. 3-month rolling gas average

### Confidence intervals
Intervals are estimated from empirical training residuals (10th–90th percentile for the 1-month model; 7.5th–92.5th for the 12-month model). This is more stable than quantile regression on datasets of this size.

### Forecast output
- **Months 1–3**: weekly point estimate + confidence interval (1-month model)
- **Months 4–12**: monthly point estimate + confidence interval (12-month model)
- All forecasts assume current market conditions persist. A gas forward curve would significantly improve the price path.

---

## Output

After running the full pipeline, outputs are saved to `data/processed/`:

| File | Description |
|------|-------------|
| `model_features.csv` | Monthly aligned feature matrix |
| `model_features_weekly.csv` | Weekly interpolated features |
| `forecast_output.csv` | Full forecast table (22 rows) |
| `forecast_summary.txt` | Human-readable forecast table |
| `forecast_dashboard.html` | Interactive browser dashboard |
| `eval_results_1m.csv` | Test set predictions vs actuals |
| `walkforward_1m.csv` | Walk-forward backtest results |

Open `forecast_dashboard.html` directly in Chrome or Edge — no server required.

---

## Known Limitations

- **Flat forecast**: all periods return the same price level because the model uses current market conditions as input for every future date. A gas price forward curve would enable a true price path.
- **GIE gas storage**: full history requires the GIE API key. Without it, only ~10 months of storage data are available, so gas storage features are excluded from the model.
- **Training data**: ~70 rows for the 1-month model is thin. Model reliability improves significantly with more AHDB history or a higher-frequency target variable.
- **12-month model**: walk-forward MAPE of ~20% reflects the inherent difficulty of 12-month commodity forecasting on this dataset size. Use as a range indicator, not a point forecast.
- **No supply shock handling**: the model missed the March 2026 AN price spike (+£130/tonne). Exogenous supply disruptions cannot be forecast from lagged features alone.

---

## Roadmap (v2 improvements)

- [ ] Integrate NBP gas forward curve for a true forecast price path
- [ ] Resolve GIE full history via authenticated API key
- [ ] Add HMRC UK fertiliser import volumes (HS code 3102)
- [ ] Extend AHDB history back to 2014 if available
- [ ] Separate short-term model (weekly, 0–12 weeks) from long-term (monthly, 3–12 months)
- [ ] Scheduled weekly refresh via cron or Task Scheduler
- [ ] Add scenario analysis: low/base/high gas price scenarios

---

## Architecture Notes

**Lookahead bias prevention**: all features are lagged so that at prediction time *t*, every feature value was observable at *t-1* or earlier. Train/test split is a contiguous time block (not random), with the last 12 months held out as test.

**Data vintage tracking**: every database row is stamped with both `data_date` (what the data refers to) and `retrieved_at` (when it was pulled). This distinction is critical for honest backtesting.

**Deduplication**: all database writes use `INSERT OR IGNORE` on unique constraints, so re-running ingestion is safe and idempotent.

---

## License

Internal PoC — not for distribution.
