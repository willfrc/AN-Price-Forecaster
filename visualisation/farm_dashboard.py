"""
Farm procurement dashboard — Phase 4.

Generates a self-contained HTML page that:
    - Takes farm inputs via a form
    - Runs the full Phase 1 + 2 + 3 pipeline
    - Displays the procurement plan with charts

Two modes:
    1. STATIC: py visualisation/farm_dashboard.py --postcode PE1 --crop winter_wheat ...
       Generates a one-off HTML file for given inputs.

    2. SERVER: py visualisation/farm_dashboard.py --serve
       Starts a lightweight local HTTP server with a live input form.
       Open http://localhost:8080 in your browser.

Output (static mode): data/processed/farm_dashboard.html
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import http.server
import urllib.parse
from datetime import date, datetime

import pandas as pd

from demand.n_calculator import calculate_n_requirement, SUPPORTED_CROPS
from demand.weather_adjuster import adjust_for_weather
from demand.procurement_planner import build_procurement_plan, ProcurementPlan
from utils.logger import get_logger

logger = get_logger(__name__)
OUTPUT_DIR = "data/processed"


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

def render_plan_html(plan: ProcurementPlan, inputs: dict) -> str:
    """Renders the procurement plan as a full self-contained HTML page."""

    # Build chart data
    months      = [r.recommended_buy_month_name[:3] for r in plan.recommendations]
    tonnes      = [r.an_tonnes for r in plan.recommendations]
    costs_pt    = [r.cost_point for r in plan.recommendations]
    costs_lo    = [r.cost_lower for r in plan.recommendations]
    costs_hi    = [r.cost_upper for r in plan.recommendations]
    prices_pt   = [r.forecast_price_point for r in plan.recommendations]
    prices_lo   = [r.forecast_price_lower for r in plan.recommendations]
    prices_hi   = [r.forecast_price_upper for r in plan.recommendations]
    actions     = [r.action for r in plan.recommendations]
    labels      = [r.application.original.label for r in plan.recommendations]

    action_colours = {
        "BUY FORWARD":        "#3fb950",
        "BUY 4-6 WEEKS AHEAD": "#58a6ff",
        "WAIT — BUY SPOT":    "#d29922",
    }
    bar_colours = json.dumps([action_colours.get(a, "#8b949e") for a in actions])

    generated = datetime.now().strftime("%d %B %Y, %H:%M")
    nvz_badge = (
        '<span class="badge badge-red">IN NVZ</span>'
        if plan.recommendations and plan.recommendations[0].application.original.nvz_restricted
        else '<span class="badge badge-grey">Non-NVZ</span>'
    )

    # Table rows
    table_rows = ""
    for rec in plan.recommendations:
        action_cls = {
            "BUY FORWARD":        "action-green",
            "BUY 4-6 WEEKS AHEAD": "action-blue",
            "WAIT — BUY SPOT":    "action-amber",
        }.get(rec.action, "")
        adj_note = (
            f'<span class="shifted">Adjusted from {rec.application.original.month_name[:3]}</span>'
            if rec.application.timing_changed else ""
        )
        table_rows += f"""
        <tr>
          <td>{rec.recommended_buy_month_name} {_buy_year(rec)}</td>
          <td>{rec.application.original.label}{adj_note}</td>
          <td class="num">{rec.an_tonnes:.1f}t</td>
          <td class="num">£{rec.forecast_price_point:.0f}</td>
          <td class="num muted">£{rec.forecast_price_lower:.0f}–£{rec.forecast_price_upper:.0f}</td>
          <td class="num">£{rec.cost_point:,.0f}</td>
          <td class="num muted">£{rec.cost_lower:,.0f}–£{rec.cost_upper:,.0f}</td>
          <td><span class="action-badge {action_cls}">{rec.action}</span></td>
        </tr>"""

    # Warning rows
    all_notes = plan.notes[:]
    for rec in plan.recommendations:
        all_notes.extend(rec.application.notes)
    notes_html = ""
    if all_notes:
        notes_html = '<div class="notes-box"><ul>' + \
            "".join(f"<li>{n}</li>" for n in all_notes[:8]) + \
            "</ul></div>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Farm Procurement Plan — {plan.farm_postcode}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Georgia', serif; background: #0d1117; color: #e6edf3; }}

  .header {{
    background: #0d1117; border-bottom: 1px solid #21262d;
    padding: 20px 36px; display: flex; justify-content: space-between; align-items: flex-end;
  }}
  .header h1 {{ font-size: 13px; font-family: 'Courier New', monospace; color: #7d8590;
    letter-spacing: 0.1em; text-transform: uppercase; }}
  .header h2 {{ font-size: 26px; font-weight: 700; color: #e6edf3; margin-top: 4px; }}
  .header-right {{ font-family: 'Courier New', monospace; font-size: 10px;
    color: #484f58; text-align: right; line-height: 1.9; }}

  .main {{ padding: 28px 36px; }}

  /* Farm summary bar */
  .farm-bar {{
    background: #161b22; border: 1px solid #21262d; border-radius: 6px;
    padding: 14px 20px; margin-bottom: 24px;
    display: flex; gap: 32px; align-items: center; flex-wrap: wrap;
  }}
  .farm-field {{ display: flex; flex-direction: column; gap: 2px; }}
  .farm-field-label {{ font-family: 'Courier New', monospace; font-size: 9px;
    color: #484f58; letter-spacing: 0.1em; text-transform: uppercase; }}
  .farm-field-value {{ font-size: 14px; color: #e6edf3; }}

  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 3px;
    font-family: 'Courier New', monospace; font-size: 10px; }}
  .badge-red    {{ background: rgba(248,81,73,0.15); color: #f85149; }}
  .badge-grey   {{ background: rgba(139,148,158,0.15); color: #8b949e; }}

  /* KPI row */
  .kpi-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }}
  .kpi {{
    background: #161b22; border: 1px solid #21262d; border-radius: 6px;
    padding: 18px 20px; position: relative; overflow: hidden;
  }}
  .kpi::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }}
  .kpi.tonnes::before {{ background: #58a6ff; }}
  .kpi.cost::before   {{ background: #3fb950; }}
  .kpi.apps::before   {{ background: #d29922; }}
  .kpi-label {{ font-family: 'Courier New', monospace; font-size: 9px;
    letter-spacing: 0.12em; text-transform: uppercase; color: #7d8590; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 34px; font-weight: 700; color: #e6edf3; line-height: 1; }}
  .kpi-value span {{ font-size: 16px; font-weight: 400; color: #7d8590; }}
  .kpi-sub {{ font-size: 11px; color: #7d8590; margin-top: 4px; }}

  /* Charts */
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
  .chart-panel {{
    background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 20px;
  }}
  .chart-title {{ font-family: 'Courier New', monospace; font-size: 10px;
    letter-spacing: 0.1em; text-transform: uppercase; color: #7d8590; margin-bottom: 16px; }}
  .chart-container {{ position: relative; height: 220px; }}

  /* Table */
  .table-panel {{ background: #161b22; border: 1px solid #21262d;
    border-radius: 6px; padding: 20px; margin-bottom: 20px; }}
  .table-title {{ font-family: 'Courier New', monospace; font-size: 10px;
    letter-spacing: 0.1em; text-transform: uppercase; color: #7d8590; margin-bottom: 14px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ font-family: 'Courier New', monospace; font-size: 10px; color: #7d8590;
    font-weight: 400; padding: 6px 10px; border-bottom: 1px solid #21262d;
    text-align: left; letter-spacing: 0.05em; }}
  td {{ padding: 9px 10px; border-bottom: 1px solid #0d1117; color: #e6edf3; }}
  tr:last-child td {{ border-bottom: none; }}
  .num {{ text-align: right; font-family: 'Courier New', monospace; }}
  .muted {{ color: #7d8590; font-size: 11px; }}
  .shifted {{ display: block; font-size: 10px; color: #d29922;
    font-family: 'Courier New', monospace; margin-top: 2px; }}

  .action-badge {{ display: inline-block; padding: 3px 8px; border-radius: 3px;
    font-family: 'Courier New', monospace; font-size: 10px; white-space: nowrap; }}
  .action-green  {{ background: rgba(63,185,80,0.15); color: #3fb950; }}
  .action-blue   {{ background: rgba(88,166,255,0.15); color: #58a6ff; }}
  .action-amber  {{ background: rgba(210,153,34,0.15); color: #d29922; }}

  tfoot td {{ font-weight: 700; border-top: 1px solid #30363d; color: #e6edf3;
    font-family: 'Courier New', monospace; }}

  /* Notes */
  .notes-box {{ background: rgba(210,153,34,0.08); border: 1px solid rgba(210,153,34,0.2);
    border-radius: 6px; padding: 14px 18px; margin-bottom: 20px; }}
  .notes-box ul {{ list-style: none; }}
  .notes-box li {{ font-size: 12px; color: #d29922; padding: 3px 0; line-height: 1.5; }}
  .notes-box li::before {{ content: "! "; font-family: 'Courier New', monospace; }}

  /* Legend */
  .legend {{ display: flex; gap: 20px; margin-top: 8px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px;
    font-family: 'Courier New', monospace; font-size: 10px; color: #7d8590; }}
  .legend-dot {{ width: 8px; height: 8px; border-radius: 2px; }}

  /* Disclaimer */
  .disclaimer {{ font-family: 'Courier New', monospace; font-size: 10px;
    color: #484f58; line-height: 1.8; padding: 14px 36px;
    border-top: 1px solid #21262d; }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>UK Fertiliser Procurement Tool</h1>
    <h2>Seasonal AN Procurement Plan &mdash; {plan.farm_postcode}</h2>
  </div>
  <div class="header-right">
    Generated: {generated}<br>
    Crop: {plan.crop_name}<br>
    Model: XGBoost + RB209 agronomics
  </div>
</div>

<div class="main">

  <div class="farm-bar">
    <div class="farm-field">
      <span class="farm-field-label">Postcode</span>
      <span class="farm-field-value">{plan.farm_postcode}</span>
    </div>
    <div class="farm-field">
      <span class="farm-field-label">Crop</span>
      <span class="farm-field-value">{plan.crop_name}</span>
    </div>
    <div class="farm-field">
      <span class="farm-field-label">Farm size</span>
      <span class="farm-field-value">{plan.farm_size_ha:.0f} ha</span>
    </div>
    <div class="farm-field">
      <span class="farm-field-label">Soil quality</span>
      <span class="farm-field-value">{inputs.get('soil_quality', '?')}/5</span>
    </div>
    <div class="farm-field">
      <span class="farm-field-label">Planting</span>
      <span class="farm-field-value">{inputs.get('planting_month_name', '?')}</span>
    </div>
    <div class="farm-field">
      <span class="farm-field-label">NVZ</span>
      <span class="farm-field-value">{nvz_badge}</span>
    </div>
  </div>

  <div class="kpi-row">
    <div class="kpi tonnes">
      <div class="kpi-label">Total AN required</div>
      <div class="kpi-value">{plan.total_an_tonnes:.1f}<span> tonnes</span></div>
      <div class="kpi-sub">34.5%N bulk ammonium nitrate</div>
    </div>
    <div class="kpi cost">
      <div class="kpi-label">Estimated seasonal cost</div>
      <div class="kpi-value">£{plan.total_cost_point:,.0f}</div>
      <div class="kpi-sub">Range: £{plan.total_cost_lower:,.0f} – £{plan.total_cost_upper:,.0f}</div>
    </div>
    <div class="kpi apps">
      <div class="kpi-label">Applications</div>
      <div class="kpi-value">{len(plan.recommendations)}<span> splits</span></div>
      <div class="kpi-sub">Weather-adjusted timing</div>
    </div>
  </div>

  <div class="chart-row">
    <div class="chart-panel">
      <div class="chart-title">Procurement schedule — tonnes by buy month</div>
      <div class="chart-container"><canvas id="tonnesChart"></canvas></div>
    </div>
    <div class="chart-panel">
      <div class="chart-title">Forecast AN price at purchase time (£/tonne)</div>
      <div class="chart-container"><canvas id="priceChart"></canvas></div>
      <div class="legend">
        <div class="legend-item">
          <div class="legend-dot" style="background:#58a6ff"></div>Point forecast
        </div>
        <div class="legend-item">
          <div class="legend-dot" style="background:#58a6ff;opacity:0.3"></div>Confidence interval
        </div>
      </div>
    </div>
  </div>

  {notes_html}

  <div class="table-panel">
    <div class="table-title">Full procurement schedule</div>
    <table>
      <thead>
        <tr>
          <th>Buy Month</th>
          <th>Application</th>
          <th>Tonnes</th>
          <th>£/t (point)</th>
          <th>£/t range</th>
          <th>Cost (point)</th>
          <th>Cost range</th>
          <th>Recommendation</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
      <tfoot>
        <tr>
          <td colspan="2">TOTAL</td>
          <td class="num">{plan.total_an_tonnes:.1f}t</td>
          <td></td><td></td>
          <td class="num">£{plan.total_cost_point:,.0f}</td>
          <td class="num muted">£{plan.total_cost_lower:,.0f}–£{plan.total_cost_upper:,.0f}</td>
          <td></td>
        </tr>
      </tfoot>
    </table>
  </div>

</div>

<div class="disclaimer">
  Price forecasts are XGBoost model outputs, not guaranteed prices. &nbsp;|&nbsp;
  1-month model MAPE ~7% &nbsp;|&nbsp; 12-month model MAPE ~11.6% &nbsp;|&nbsp;
  N requirements from AHDB RB209. &nbsp;|&nbsp;
  Weather data from Open-Meteo (open-meteo.com). &nbsp;|&nbsp;
  This tool does not constitute financial or agronomic advice.
  Consult a BASIS-qualified agronomist before applying nitrogen.
</div>

<script>
const months   = {json.dumps(months)};
const tonnes   = {json.dumps(tonnes)};
const costsLo  = {json.dumps(costs_lo)};
const costsHi  = {json.dumps(costs_hi)};
const pricesLo = {json.dumps(prices_lo)};
const pricesHi = {json.dumps(prices_hi)};
const pricesPt = {json.dumps(prices_pt)};
const barColours = {bar_colours};

// Tonnes chart
new Chart(document.getElementById('tonnesChart'), {{
  type: 'bar',
  data: {{
    labels: months,
    datasets: [{{
      label: 'AN tonnes',
      data: tonnes,
      backgroundColor: barColours,
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        backgroundColor: '#1c2128', borderColor: '#30363d', borderWidth: 1,
        titleColor: '#7d8590', bodyColor: '#e6edf3',
        callbacks: {{ label: ctx => ` ${{ctx.parsed.y.toFixed(1)}} tonnes` }}
      }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#7d8590', font: {{ family: 'Courier New', size: 10 }} }},
             grid: {{ color: '#21262d' }} }},
      y: {{ ticks: {{ color: '#7d8590', font: {{ family: 'Courier New', size: 10 }},
                      callback: v => v + 't' }},
             grid: {{ color: '#21262d' }} }}
    }}
  }}
}});

// Price chart with CI band
new Chart(document.getElementById('priceChart'), {{
  type: 'line',
  data: {{
    labels: months,
    datasets: [
      {{
        label: 'Upper bound',
        data: pricesHi,
        borderWidth: 0,
        backgroundColor: 'rgba(88,166,255,0.15)',
        fill: '+1',
        pointRadius: 0,
        tension: 0.3,
      }},
      {{
        label: 'Lower bound',
        data: pricesLo,
        borderWidth: 0,
        backgroundColor: 'rgba(88,166,255,0.15)',
        fill: false,
        pointRadius: 0,
        tension: 0.3,
      }},
      {{
        label: 'Point forecast',
        data: pricesPt,
        borderColor: '#58a6ff',
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointBackgroundColor: '#58a6ff',
        pointRadius: 5,
        tension: 0.3,
      }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        backgroundColor: '#1c2128', borderColor: '#30363d', borderWidth: 1,
        titleColor: '#7d8590', bodyColor: '#e6edf3',
        callbacks: {{
          label: ctx => {{
            if (ctx.dataset.label.includes('bound')) return null;
            return ` £${{ctx.parsed.y.toFixed(0)}}/t`;
          }}
        }}
      }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#7d8590', font: {{ family: 'Courier New', size: 10 }} }},
             grid: {{ color: '#21262d' }} }},
      y: {{ ticks: {{ color: '#7d8590', font: {{ family: 'Courier New', size: 10 }},
                      callback: v => '£' + v }},
             grid: {{ color: '#21262d' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Static generator
# ---------------------------------------------------------------------------

def generate_static(crop, farm_size_ha, soil_quality, planting_month,
                    postcode, nvz_override=None):
    """Runs the full pipeline and writes the HTML file."""

    MONTH_NAMES = {
        1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
        7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"
    }

    print(f"\nGenerating procurement plan for {postcode}...")
    print(f"  Crop: {SUPPORTED_CROPS.get(crop, crop)}, {farm_size_ha}ha, "
          f"soil quality {soil_quality}/5, planting {MONTH_NAMES[planting_month]}")

    farm     = calculate_n_requirement(
        crop=crop, farm_size_ha=farm_size_ha, soil_quality=soil_quality,
        planting_month=planting_month, postcode=postcode,
        nvz_override=nvz_override,
    )
    adjusted = adjust_for_weather(farm)
    plan     = build_procurement_plan(adjusted)

    inputs = {
        "soil_quality": soil_quality,
        "planting_month_name": MONTH_NAMES[planting_month],
    }

    html = render_plan_html(plan, inputs)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "farm_dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nDashboard saved: {out_path}")
    print("Open this file in your browser.")
    print(plan.summary())
    return plan


# ---------------------------------------------------------------------------
# Live server mode
# ---------------------------------------------------------------------------

FORM_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Farm Procurement Tool</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Georgia', serif; background: #0d1117; color: #e6edf3;
    display: flex; justify-content: center; padding: 48px 20px; }}
  .card {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px;
    padding: 36px 40px; width: 100%; max-width: 520px; }}
  h1 {{ font-size: 11px; font-family: 'Courier New', monospace; color: #7d8590;
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 4px; }}
  h2 {{ font-size: 22px; font-weight: 700; margin-bottom: 28px; }}
  .field {{ margin-bottom: 18px; }}
  label {{ display: block; font-family: 'Courier New', monospace; font-size: 10px;
    color: #7d8590; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px; }}
  input, select {{ width: 100%; background: #0d1117; border: 1px solid #30363d;
    border-radius: 4px; padding: 10px 12px; color: #e6edf3; font-size: 14px;
    font-family: Georgia, serif; outline: none; }}
  input:focus, select:focus {{ border-color: #58a6ff; }}
  select option {{ background: #161b22; }}
  .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  button {{ width: 100%; background: #238636; border: none; border-radius: 4px;
    color: #fff; font-size: 14px; font-family: Georgia, serif; padding: 12px;
    cursor: pointer; margin-top: 8px; letter-spacing: 0.02em; }}
  button:hover {{ background: #2ea043; }}
  .note {{ font-size: 11px; color: #484f58; margin-top: 16px;
    font-family: 'Courier New', monospace; line-height: 1.7; }}
</style>
</head>
<body>
<div class="card">
  <h1>UK Fertiliser Procurement Tool</h1>
  <h2>Farm Details</h2>
  <form method="GET" action="/plan">
    <div class="field">
      <label>UK Postcode</label>
      <input name="postcode" placeholder="e.g. PE1 1AB" required>
    </div>
    <div class="field">
      <label>Crop</label>
      <select name="crop">
        <option value="winter_wheat">Winter Wheat</option>
        <option value="spring_barley">Spring Barley</option>
        <option value="oilseed_rape">Oilseed Rape (Winter)</option>
        <option value="maize">Maize</option>
      </select>
    </div>
    <div class="row">
      <div class="field">
        <label>Farm Size (ha)</label>
        <input name="size" type="number" min="1" max="10000" placeholder="e.g. 120" required>
      </div>
      <div class="field">
        <label>Soil Quality (1–5)</label>
        <select name="soil">
          <option value="1">1 — Poor (sandy, low OM)</option>
          <option value="2">2 — Below average</option>
          <option value="3" selected>3 — Average</option>
          <option value="4">4 — Good</option>
          <option value="5">5 — Excellent (high OM)</option>
        </select>
      </div>
    </div>
    <div class="field">
      <label>Planting Month</label>
      <select name="month">
        <option value="1">January</option>
        <option value="2">February</option>
        <option value="3">March</option>
        <option value="4">April</option>
        <option value="5">May</option>
        <option value="6">June</option>
        <option value="7">July</option>
        <option value="8">August</option>
        <option value="9" selected>September</option>
        <option value="10">October</option>
        <option value="11">November</option>
        <option value="12">December</option>
      </select>
    </div>
    <button type="submit">Generate Procurement Plan</button>
  </form>
  <p class="note">
    Plan generated in ~10 seconds. Includes weather-adjusted application timing,<br>
    AN price forecast, and buy timing recommendations with cost estimates.
  </p>
</div>
</body>
</html>"""


class FarmPlanHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP handler for the farm dashboard server."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/":
            self._send_html(FORM_HTML)

        elif parsed.path == "/plan":
            params = urllib.parse.parse_qs(parsed.query)
            try:
                crop        = params.get("crop", ["winter_wheat"])[0]
                postcode    = params.get("postcode", ["PE1 1AB"])[0]
                size        = float(params.get("size", ["100"])[0])
                soil        = int(params.get("soil", ["3"])[0])
                month       = int(params.get("month", ["9"])[0])

                MONTH_NAMES = {
                    1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                    7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"
                }

                farm     = calculate_n_requirement(
                    crop=crop, farm_size_ha=size, soil_quality=soil,
                    planting_month=month, postcode=postcode,
                )
                adjusted = adjust_for_weather(farm)
                plan     = build_procurement_plan(adjusted)
                inputs   = {"soil_quality": soil, "planting_month_name": MONTH_NAMES[month]}
                html     = render_plan_html(plan, inputs)
                self._send_html(html)

            except Exception as e:
                self._send_html(
                    f"<html><body style='background:#0d1117;color:#f85149;"
                    f"font-family:monospace;padding:40px'>"
                    f"<h2>Error generating plan</h2><pre>{e}</pre>"
                    f"<a href='/' style='color:#58a6ff'>Try again</a></body></html>"
                )
        else:
            self.send_response(404)
            self.end_headers()

    def _send_html(self, html: str):
        encoded = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(encoded))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        logger.info(f"HTTP {args[0]} {args[1]}")


def run_server(port: int = 8080):
    server = http.server.HTTPServer(("localhost", port), FarmPlanHandler)
    print(f"\nFarm Procurement Tool running at: http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buy_year(rec) -> int:
    current_month = date.today().month
    return date.today().year if rec.recommended_buy_month >= current_month else date.today().year + 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Farm AN Procurement Dashboard")
    parser.add_argument("--serve",    action="store_true", help="Start live server at localhost:8080")
    parser.add_argument("--port",     type=int, default=8080)
    parser.add_argument("--postcode", default="PE1 1AB")
    parser.add_argument("--crop",     default="winter_wheat", choices=list(SUPPORTED_CROPS.keys()))
    parser.add_argument("--size",     type=float, default=120.0, help="Farm size in ha")
    parser.add_argument("--soil",     type=int,   default=3, choices=[1,2,3,4,5])
    parser.add_argument("--month",    type=int,   default=9, help="Planting month (1-12)")
    parser.add_argument("--nvz",      type=lambda x: x.lower() == "true",
                        default=None, help="NVZ override: true/false")
    args = parser.parse_args()

    if args.serve:
        run_server(args.port)
    else:
        generate_static(
            crop=args.crop,
            farm_size_ha=args.size,
            soil_quality=args.soil,
            planting_month=args.month,
            postcode=args.postcode,
            nvz_override=args.nvz,
        )
