# UK Ammonium Nitrate Price Forecaster — Data Ingestion Pipeline

## Project Structure

```
an_price_forecaster/
├── ingestion/
│   ├── fred_ingestion.py        # FRED API: gas, FX, Brent, urea
│   ├── yfinance_ingestion.py    # Equity proxies, CME urea futures
│   ├── gie_ingestion.py         # EU gas storage (GIE AGSI+)
│   ├── worldbank_ingestion.py   # World Bank Pink Sheet (urea, DAP, ammonia)
│   ├── ahdb_ingestion.py        # AHDB UK AN spot price (manual CSV loader)
│   └── ember_ingestion.py       # EU ETS carbon price
├── utils/
│   ├── db.py                    # SQLite database manager
│   ├── logger.py                # Logging setup
│   └── config.py                # Central config (dates, tickers, series IDs)
├── data/
│   ├── raw/                     # Raw pulls, never modified
│   └── processed/               # Aligned, cleaned data
├── run_ingestion.py             # Master script: runs all ingestion modules
├── requirements.txt
└── .env.example
```

## Setup

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API keys
```bash
cp .env.example .env
# Edit .env and add your FRED API key
```
Get a free FRED API key at: https://fredaccount.stlouisfed.org/apikeys

### 4. AHDB manual data
- Download the UK fertiliser price series from: https://ahdb.org.uk/dairy/uk-fertiliser-price-tracker
- Save as `data/raw/ahdb_an_prices.csv`
- The loader expects columns: `Date`, `Price_GBP_tonne`

### 5. Run the pipeline
```bash
python run_ingestion.py
```

## Data Sources

| Source | Series | Frequency | Free |
|--------|--------|-----------|------|
| FRED | NBP gas, TTF gas, Brent, USD/GBP, urea | Monthly/Daily | Yes (API key) |
| yfinance | SOIL ETF, CF Industries, Yara, CME urea futures | Daily | Yes |
| GIE AGSI+ | EU gas storage levels | Daily | Yes |
| World Bank | Urea, DAP, ammonia prices | Monthly | Yes |
| AHDB | UK AN spot price | Weekly | Yes (manual) |
| Ember Climate | EU ETS carbon price | Weekly | Yes (manual) |

## Notes on Limitations
- AHDB and Ember data require manual CSV downloads (no public API)
- Best-in-class UK AN price data (ICIS, Argus, CRU) is paywalled — upgrade for production use
- FRED gas series are monthly; GIE AGSI+ provides daily gas storage as a higher-frequency proxy
- CME urea futures via yfinance are directional only — treat as sentiment signal, not benchmark
