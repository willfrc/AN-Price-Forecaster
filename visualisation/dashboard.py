"""
Interactive forecast dashboard.
Generates a self-contained HTML file with Chart.js visualisations.
No server required — open in any browser.

Run: py visualisation/dashboard.py
Output: data/processed/forecast_dashboard.html
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_DIR  = "data/processed"
SAVED_DIR   = "model/saved"


def build_dashboard():
    # Load data
    forecast = _load_forecast()
    actuals  = _load_actuals()
    metrics  = _load_metrics()

    if forecast is None:
        logger.error("Forecast data not found. Run: py model/forecast.py")
        return

    html = _render_html(forecast, actuals, metrics)

    out_path = os.path.join(OUTPUT_DIR, "forecast_dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Dashboard saved: {out_path}")
    print(f"\nDashboard ready: {out_path}")
    print("Open this file in your browser.")


def _load_forecast():
    path = os.path.join(OUTPUT_DIR, "forecast_output.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def _load_actuals():
    path = os.path.join(OUTPUT_DIR, "model_features.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "an_price_gbp_t" not in df.columns:
        return None
    actuals = df[["an_price_gbp_t"]].dropna().reset_index()
    actuals.columns = ["date", "price"]
    actuals = actuals[actuals["date"] >= "2021-01-01"]  # Last 4 years for context
    return actuals


def _load_metrics():
    path = os.path.join(SAVED_DIR, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _render_html(forecast, actuals, metrics):
    # Prepare chart data
    actual_labels = actuals["date"].dt.strftime("%Y-%m-%d").tolist() if actuals is not None else []
    actual_values = actuals["price"].round(1).tolist() if actuals is not None else []

    # Separate short-term (weekly) and long-term (monthly) forecasts
    weekly  = forecast[forecast["frequency"] == "weekly"]
    monthly = forecast[forecast["frequency"] == "monthly"]

    # For the chart, use one point per month from weekly (first week of each month)
    weekly_monthly = weekly.groupby(weekly["date"].dt.to_period("M")).first().reset_index(drop=True)
    forecast_chart = pd.concat([weekly_monthly, monthly]).sort_values("date")

    fc_labels = forecast_chart["date"].astype(str).str[:10].tolist()
    fc_points = forecast_chart["point_forecast"].round(1).tolist()
    fc_lower  = forecast_chart["lower_bound"].round(1).tolist()
    fc_upper  = forecast_chart["upper_bound"].round(1).tolist()

    # Latest actual for headline
    latest_price = actuals["price"].iloc[-1] if actuals is not None and len(actuals) > 0 else 0
    latest_date  = actuals["date"].iloc[-1].strftime("%b %Y") if actuals is not None and len(actuals) > 0 else "N/A"

    # Next month forecast
    next_month_point = forecast_chart["point_forecast"].iloc[0] if len(forecast_chart) > 0 else 0
    next_month_lower = forecast_chart["lower_bound"].iloc[0] if len(forecast_chart) > 0 else 0
    next_month_upper = forecast_chart["upper_bound"].iloc[0] if len(forecast_chart) > 0 else 0
    next_month_date  = fc_labels[0][:7] if fc_labels else "N/A"

    # 12-month forecast (last monthly point)
    last_point = forecast_chart["point_forecast"].iloc[-1] if len(forecast_chart) > 0 else 0
    last_lower = forecast_chart["lower_bound"].iloc[-1] if len(forecast_chart) > 0 else 0
    last_upper = forecast_chart["upper_bound"].iloc[-1] if len(forecast_chart) > 0 else 0
    last_date  = fc_labels[-1][:7] if fc_labels else "N/A"

    # Metrics
    m1  = metrics.get("horizon_1m",  {})
    m12 = metrics.get("horizon_12m", {})
    mape_1m  = m1.get("xgboost",  {}).get("mape",  "N/A")
    mape_12m = m12.get("xgboost", {}).get("mape",  "N/A")
    mae_1m   = m1.get("xgboost",  {}).get("mae",   "N/A")
    ci_1m    = m1.get("ci_label",  "80% CI")
    ci_12m   = m12.get("ci_label", "85% CI")
    ci_cov_1m  = m1.get("ci_coverage",  "N/A")
    ci_cov_12m = m12.get("ci_coverage", "N/A")
    mae_12m = m12.get("xgboost", {}).get("mae", "N/A")

    generated = datetime.now().strftime("%d %B %Y, %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UK AN Price Forecast</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Georgia', serif;
    background: #0d1117;
    color: #e6edf3;
    min-height: 100vh;
  }}

  .header {{
    background: #0d1117;
    border-bottom: 1px solid #21262d;
    padding: 24px 40px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }}

  .header-left h1 {{
    font-size: 20px;
    font-weight: 400;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7d8590;
    font-family: 'Courier New', monospace;
  }}

  .header-left h2 {{
    font-size: 32px;
    font-weight: 700;
    color: #e6edf3;
    margin-top: 4px;
    letter-spacing: -0.02em;
  }}

  .header-right {{
    text-align: right;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    color: #7d8590;
    line-height: 1.8;
  }}

  .main {{ padding: 32px 40px; }}

  /* KPI cards */
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 32px;
  }}

  .kpi {{
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
  }}

  .kpi::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }}

  .kpi.current::before  {{ background: #58a6ff; }}
  .kpi.next::before     {{ background: #3fb950; }}
  .kpi.longterm::before {{ background: #d29922; }}

  .kpi-label {{
    font-family: 'Courier New', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7d8590;
    margin-bottom: 8px;
  }}

  .kpi-value {{
    font-size: 42px;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1;
    color: #e6edf3;
  }}

  .kpi-value span {{
    font-size: 18px;
    font-weight: 400;
    color: #7d8590;
    margin-left: 2px;
  }}

  .kpi-range {{
    font-family: 'Courier New', monospace;
    font-size: 11px;
    color: #7d8590;
    margin-top: 6px;
  }}

  .kpi-sub {{
    font-size: 12px;
    color: #7d8590;
    margin-top: 4px;
  }}

  /* Chart panel */
  .chart-panel {{
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 24px;
    margin-bottom: 24px;
  }}

  .panel-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }}

  .panel-title {{
    font-family: 'Courier New', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7d8590;
  }}

  .legend {{
    display: flex;
    gap: 20px;
    font-family: 'Courier New', monospace;
    font-size: 10px;
    color: #7d8590;
  }}

  .legend-item {{ display: flex; align-items: center; gap: 6px; }}

  .legend-dot {{
    width: 8px; height: 8px; border-radius: 50%;
  }}

  .legend-band {{
    width: 16px; height: 6px; border-radius: 2px; opacity: 0.4;
  }}

  .chart-container {{ position: relative; height: 320px; }}

  /* Metrics table */
  .metrics-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
  }}

  .metrics-panel {{
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 20px 24px;
  }}

  .metrics-panel h3 {{
    font-family: 'Courier New', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7d8590;
    margin-bottom: 14px;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}

  th {{
    text-align: left;
    padding: 6px 0;
    border-bottom: 1px solid #21262d;
    font-family: 'Courier New', monospace;
    font-size: 10px;
    color: #7d8590;
    font-weight: 400;
    letter-spacing: 0.06em;
  }}

  td {{
    padding: 7px 0;
    border-bottom: 1px solid #161b22;
    color: #e6edf3;
  }}

  td:last-child {{ text-align: right; font-family: 'Courier New', monospace; }}

  .good  {{ color: #3fb950; }}
  .warn  {{ color: #d29922; }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 10px;
    letter-spacing: 0.06em;
  }}

  .badge-blue   {{ background: rgba(88,166,255,0.15); color: #58a6ff; }}
  .badge-orange {{ background: rgba(210,153,34,0.15); color: #d29922; }}

  /* Footer */
  .footer {{
    padding: 16px 40px;
    border-top: 1px solid #21262d;
    font-family: 'Courier New', monospace;
    font-size: 10px;
    color: #484f58;
    line-height: 1.9;
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <h1>UK Fertiliser Intelligence</h1>
    <h2>Ammonium Nitrate Price Forecast</h2>
  </div>
  <div class="header-right">
    Generated: {generated}<br>
    Source: AHDB UK AN bulk 34.5%N<br>
    Model: XGBoost gradient boosting
  </div>
</div>

<div class="main">

  <div class="kpi-row">
    <div class="kpi current">
      <div class="kpi-label">Latest actual &mdash; {latest_date}</div>
      <div class="kpi-value">£{latest_price:.0f}<span>/t</span></div>
      <div class="kpi-sub">AHDB UK AN bulk (34.5%N)</div>
    </div>
    <div class="kpi next">
      <div class="kpi-label">1-month forecast &mdash; {next_month_date}</div>
      <div class="kpi-value">£{next_month_point:.0f}<span>/t</span></div>
      <div class="kpi-range">Range: £{next_month_lower:.0f} &ndash; £{next_month_upper:.0f} &nbsp;|&nbsp; {ci_1m}</div>
    </div>
    <div class="kpi longterm">
      <div class="kpi-label">12-month forecast &mdash; {last_date}</div>
      <div class="kpi-value">£{last_point:.0f}<span>/t</span></div>
      <div class="kpi-range">Range: £{last_lower:.0f} &ndash; £{last_upper:.0f} &nbsp;|&nbsp; {ci_12m}</div>
    </div>
  </div>

  <div class="chart-panel">
    <div class="panel-header">
      <div class="panel-title">Price history &amp; 12-month forecast</div>
      <div class="legend">
        <div class="legend-item">
          <div class="legend-dot" style="background:#58a6ff"></div>
          <span>Actual (AHDB)</span>
        </div>
        <div class="legend-item">
          <div class="legend-dot" style="background:#3fb950"></div>
          <span>Forecast (1-month model)</span>
        </div>
        <div class="legend-item">
          <div class="legend-dot" style="background:#d29922"></div>
          <span>Forecast (12-month model)</span>
        </div>
        <div class="legend-item">
          <div class="legend-band" style="background:#3fb950"></div>
          <span>Confidence interval</span>
        </div>
      </div>
    </div>
    <div class="chart-container">
      <canvas id="mainChart"></canvas>
    </div>
  </div>

  <div class="metrics-row">
    <div class="metrics-panel">
      <h3>Model performance &mdash; test set (last 12 months)</h3>
      <table>
        <tr>
          <th>Metric</th>
          <th>1-month model</th>
          <th>12-month model</th>
        </tr>
        <tr>
          <td>MAE</td>
          <td><span class="good">£{mae_1m}/tonne</span></td>
          <td>£{mae_12m}/tonne</td>
        </tr>
        <tr>
          <td>MAPE</td>
          <td><span class="good">{mape_1m}%</span></td>
          <td><span class="warn">{mape_12m}%</span></td>
        </tr>
        <tr>
          <td>CI coverage</td>
          <td>{ci_cov_1m}%</td>
          <td>{ci_cov_12m}%</td>
        </tr>
        <tr>
          <td>CI method</td>
          <td colspan="2" style="font-family:'Courier New',monospace;font-size:11px;color:#7d8590">Empirical training residuals (not quantile regression)</td>
        </tr>
      </table>
    </div>

    <div class="metrics-panel">
      <h3>Data sources</h3>
      <table>
        <tr><th>Source</th><th>Series</th><th>Status</th></tr>
        <tr>
          <td>AHDB</td>
          <td>UK AN spot price (target)</td>
          <td><span class="badge badge-blue">LIVE</span></td>
        </tr>
        <tr>
          <td>FRED</td>
          <td>TTF gas, Brent, USD/GBP</td>
          <td><span class="badge badge-blue">LIVE</span></td>
        </tr>
        <tr>
          <td>World Bank</td>
          <td>Urea, DAP, European gas</td>
          <td><span class="badge badge-blue">LIVE</span></td>
        </tr>
        <tr>
          <td>yfinance</td>
          <td>CF Industries, Yara, SOIL</td>
          <td><span class="badge badge-blue">LIVE</span></td>
        </tr>
        <tr>
          <td>GIE AGSI+</td>
          <td>EU gas storage</td>
          <td><span class="badge badge-orange">PARTIAL</span></td>
        </tr>
      </table>
    </div>
  </div>

</div>

<div class="footer">
  Key drivers (1-month model, by importance): TTF gas price (lagged 1m) &nbsp;|&nbsp;
  Global urea price (lagged 1m) &nbsp;|&nbsp; DAP benchmark &nbsp;|&nbsp;
  Lagged AN price (1-3 months) &nbsp;|&nbsp; 3-month gas rolling average
  &nbsp;&nbsp;&bull;&nbsp;&nbsp;
  12-month model uses same features; weighted toward AN-urea spread, Brent, and 3-month momentum.
  &nbsp;&nbsp;&bull;&nbsp;&nbsp;
  Forecast assumes current market conditions persist. Does not account for supply disruptions or policy changes.
  Add NBP gas forward curve in v2 to generate a price path rather than a static level.
</div>

<script>
const actualLabels = {json.dumps(actual_labels)};
const actualValues = {json.dumps(actual_values)};
const fcLabels     = {json.dumps(fc_labels)};
const fcPoints     = {json.dumps(fc_points)};
const fcLower      = {json.dumps(fc_lower)};
const fcUpper      = {json.dumps(fc_upper)};

// Split forecast into short-term (months 1-3) and long-term (months 4-12)
// Short-term = first 3 items, long-term = rest
const fcShortPoints  = fcPoints.slice(0, 3);
const fcShortLower   = fcLower.slice(0, 3);
const fcShortUpper   = fcUpper.slice(0, 3);
const fcShortLabels  = fcLabels.slice(0, 3);
const fcLongPoints   = fcPoints.slice(3);
const fcLongLower    = fcLower.slice(3);
const fcLongUpper    = fcUpper.slice(3);
const fcLongLabels   = fcLabels.slice(3);

// Build CI band datasets (filled area between lower and upper)
const shortBandData = fcShortLabels.map((l, i) => ({{ x: l, y: fcShortUpper[i] }}));
const shortBandBase = fcShortLabels.map((l, i) => ({{ x: l, y: fcShortLower[i] }}));
const longBandData  = fcLongLabels.map((l, i) => ({{ x: l, y: fcLongUpper[i] }}));
const longBandBase  = fcLongLabels.map((l, i) => ({{ x: l, y: fcLongLower[i] }}));

// Combine all labels for x-axis
const allLabels = [...new Set([...actualLabels, ...fcLabels])].sort();

function toChartData(labels, values) {{
  const map = {{}};
  labels.forEach((l, i) => map[l] = values[i]);
  return allLabels.map(l => map[l] !== undefined ? map[l] : null);
}}

const ctx = document.getElementById('mainChart').getContext('2d');

new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: allLabels,
    datasets: [
      // Short-term CI band (upper)
      {{
        label: '_st_upper',
        data: toChartData(fcShortLabels, fcShortUpper),
        borderWidth: 0,
        backgroundColor: 'rgba(63,185,80,0.15)',
        fill: '+1',
        pointRadius: 0,
        tension: 0.3,
      }},
      // Short-term CI band (lower)
      {{
        label: '_st_lower',
        data: toChartData(fcShortLabels, fcShortLower),
        borderWidth: 0,
        backgroundColor: 'rgba(63,185,80,0.15)',
        fill: false,
        pointRadius: 0,
        tension: 0.3,
      }},
      // Long-term CI band (upper)
      {{
        label: '_lt_upper',
        data: toChartData(fcLongLabels, fcLongUpper),
        borderWidth: 0,
        backgroundColor: 'rgba(210,153,34,0.12)',
        fill: '+1',
        pointRadius: 0,
        tension: 0.3,
      }},
      // Long-term CI band (lower)
      {{
        label: '_lt_lower',
        data: toChartData(fcLongLabels, fcLongLower),
        borderWidth: 0,
        backgroundColor: 'rgba(210,153,34,0.12)',
        fill: false,
        pointRadius: 0,
        tension: 0.3,
      }},
      // Actual prices
      {{
        label: 'Actual (AHDB)',
        data: toChartData(actualLabels, actualValues),
        borderColor: '#58a6ff',
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 3,
        pointBackgroundColor: '#58a6ff',
        tension: 0.3,
      }},
      // Short-term forecast
      {{
        label: '1-month model',
        data: toChartData(fcShortLabels, fcShortPoints),
        borderColor: '#3fb950',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [5, 3],
        pointRadius: 4,
        pointBackgroundColor: '#3fb950',
        tension: 0.3,
      }},
      // Long-term forecast
      {{
        label: '12-month model',
        data: toChartData(fcLongLabels, fcLongPoints),
        borderColor: '#d29922',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [8, 4],
        pointRadius: 4,
        pointBackgroundColor: '#d29922',
        tension: 0.3,
      }},
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{
        display: true,
        labels: {{
          color: '#7d8590',
          font: {{ family: 'Courier New', size: 10 }},
          filter: item => !item.text.startsWith('_'),
          boxWidth: 12,
          boxHeight: 2,
          padding: 16,
        }}
      }},
      tooltip: {{
        backgroundColor: '#1c2128',
        borderColor: '#30363d',
        borderWidth: 1,
        titleColor: '#7d8590',
        bodyColor: '#e6edf3',
        titleFont: {{ family: 'Courier New', size: 10 }},
        bodyFont: {{ family: 'Georgia', size: 13 }},
        padding: 12,
        callbacks: {{
          label: ctx => {{
            if (ctx.dataset.label.startsWith('_')) return null;
            const v = ctx.parsed.y;
            if (v === null) return null;
            return ` ${{ctx.dataset.label}}: £${{v.toFixed(0)}}/t`;
          }}
        }}
      }}
    }},
    scales: {{
      x: {{
        type: 'category',
        ticks: {{
          color: '#7d8590',
          font: {{ family: 'Courier New', size: 9 }},
          maxTicksLimit: 18,
          maxRotation: 45,
        }},
        grid: {{ color: '#21262d' }},
      }},
      y: {{
        ticks: {{
          color: '#7d8590',
          font: {{ family: 'Courier New', size: 10 }},
          callback: v => '£' + v,
        }},
        grid: {{ color: '#21262d' }},
        title: {{
          display: true,
          text: 'GBP / tonne',
          color: '#484f58',
          font: {{ family: 'Courier New', size: 10 }},
        }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


if __name__ == "__main__":
    build_dashboard()
