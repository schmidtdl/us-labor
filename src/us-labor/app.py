import pandas as pd
import datetime as dt
import streamlit as st
import calculations as ct
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── Constants ─────────────────────────────────────────────────────────────────
COVID_START = "2020-02-01"
COVID_END = "2021-12-01"
COLORS = dict(
    teal="#01696f",
    grey="#dcd9d5",
    text="#28251d",
    muted="#7a7974",
    green_deep=(88, 166, 119),
    green_mid=(212, 237, 218),
    pink_deep=(185, 80, 150),
    pink_mid=(242, 214, 232),
    white=(255, 255, 255),
)

SECTOR_LABELS = {
    "mining": "Mining",
    "construction": "Construction",
    "manufacturing": "Manufacturing",
    "trade & utilities": "Trade & Utilities",
    "information": "Information",
    "financial services": "Financial Serv.",
    "professional services": "Professional Serv.",
    "education & healthcare": "Education & Health",
    "leisure & hospitality": "Leisure & Hospitality",
    "other services": "Other Serv.",
    "government": "Government",
}

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; background-color: #f7f6f2 !important; }
  .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
  [data-testid="collapsedControl"] { display: none; }
  .app-main-title  { font-size: 2rem; font-weight: 700; color: #28251d; letter-spacing: -0.03em; margin-bottom: 0.1rem; line-height: 1.15; }
  .page-title      { font-size: 1.5rem; font-weight: 500; color: #7a7974; letter-spacing: -0.01em; margin-bottom: 1.25rem; }
  .section-label   { font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: #28251d; margin-bottom: 0.6rem; }
  .stSelectbox label, .stNumberInput label {
    font-size: 0.7rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.05em !important; color: #7a7974 !important;
  }
  .stSelectbox > div > div,
  .stNumberInput div[data-baseweb="input"],
  .stNumberInput div[data-baseweb="input"] > div {
    background: #ffffff !important; border: 0px solid #dcd9d5 !important; border-radius: 6px !important;
  }
  hr { border-color: #dcd9d5; margin: 1rem 0; }
  .metric-card  { background: #ffffff; border: 1px solid #dcd9d5; border-radius: 8px; padding: 14px 18px; }
  .metric-label { color: #7a7974; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
  .metric-value { color: #28251d; font-size: 1.4rem; font-weight: 700; font-variant-numeric: tabular-nums; }
  .metric-sub   { color: #bab9b4; font-size: 0.72rem; margin-top: 2px; }
  .breakeven-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; font-variant-numeric: tabular-nums; }
  .breakeven-table th {
    background: #f0ede8; color: #7a7974; font-weight: 600; font-size: 0.82rem;
    text-transform: uppercase; letter-spacing: 0.03em; padding: 9px 14px;
    border-bottom: 1px solid #dcd9d5; text-align: center; white-space: nowrap;
  }
  .breakeven-table th.row-header { text-align: left; }
  .breakeven-table td { padding: 7px 12px; border-bottom: 1px solid #edeae5; text-align: center; font-weight: 500; white-space: nowrap; }
  .breakeven-table tr:last-child td { border-bottom: none; }
  .breakeven-table td.row-label { color: #7a7974; font-size: 0.85rem; text-align: left; font-weight: 500; background: #f9f8f5 !important; }
  .breakeven-table td.cell-intersection { outline: 2px solid #2d6a4f; outline-offset: -2px; font-weight: 700 !important; }
  .layoffs-row:hover td { background: #f0ede8 !important; }
  .bev-toggle label { font-size: 0.7rem !important; }
</style>
"""

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / "assets" / "raw.csv", index_col="dates")
    df.index = pd.to_datetime(df.index, dayfirst=True)
    return df.astype(float).sort_index().dropna(how="all")

@st.cache_data
def load_layoffs() -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / "assets" / "layoffs.csv")

@st.cache_data
def compute_trend(span: int = 12) -> pd.DataFrame:
    return ct.calculate_trend_growth(load_data(), span=span).iloc[-60:]

@st.cache_data
def compute_matrix(months: int, net_migration: float, trend: float):
    calc  = ct.UnemploymentCalculator(load_data())
    ratio = ct.calculate_release_ratio(load_data(), period=12)
    shared = dict(months=months, source_ratio=ratio, net_migration_growth_estimate=net_migration)
    matrix     = calc.build_breakeven_table(**shared)
    expected_ur = calc.extrapolate_unemployment_rate(**shared, trend_rate=trend)
    return matrix, calc.last_participation * 100, calc.last_unemployment_rate * 100, calc.last_date, expected_ur * 100

@st.cache_data
def compute_beveridge(include_covid: bool) -> pd.DataFrame:
    return ct.return_beveridge_curve(load_data(), include_covid=include_covid)

@st.cache_data
def compute_sectors() -> pd.DataFrame:
    return ct.UnemploymentSector(load_data()).calculate_sector_slack().dropna().iloc[-120:]

# ── Colour helpers ────────────────────────────────────────────────────────────
def lerp(c1: tuple, c2: tuple, t: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(a + (b - a) * t) for a, b in zip(c1, c2)))

def cell_colours(val: float, v_min: float, v_max: float) -> tuple[str, str]:
    C = COLORS
    if val == 0:
        return "#f0faf2", C["teal"]
    if val > 0:
        t = min(val / v_max, 1.0)
        bg = lerp(C["white"], C["green_mid"], t * 2) if t < 0.5 else lerp(C["green_mid"], C["green_deep"], (t - 0.5) * 2)
        fg = "#1a4d2e" if t > 0.55 else "#2d6a4f" if t > 0.25 else "#437a22"
    else:
        t = min(abs(val) / abs(v_min), 1.0)
        bg = lerp(C["white"], C["pink_mid"], t * 2) if t < 0.5 else lerp(C["pink_mid"], C["pink_deep"], (t - 0.5) * 2)
        fg = "#ffffff" if t > 0.6 else "#5c0d3d" if t > 0.3 else "#7d1e5e"
    return bg, fg

def signed(val: float) -> str:
    return "+" if val > 0 else ""

def value_color(val: float, pos="#32CD32", zero=None, neg="#a12c7b") -> str:
    return pos if val > 0 else (zero or COLORS["teal"]) if val == 0 else neg

# ── Shared chart layout ───────────────────────────────────────────────────────
def base_layout(height: int, **overrides) -> dict:
    C = COLORS
    layout = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=16, b=0),
        height=height,
        font=dict(family="Inter", color=C["text"], size=11),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=C["muted"], size=10),
                   showline=True, linecolor=C["grey"], linewidth=1),
        yaxis=dict(showgrid=True,  gridcolor="#edeae5", zeroline=False,
                   tickfont=dict(color=C["muted"], size=10),
                   showline=True, linecolor=C["grey"], linewidth=1),
    )

    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(layout.get(k), dict):
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    return layout

def axis(title: str, suffix: str = "", grid: bool = True, **kw) -> dict:
    C = COLORS
    return dict(
        title=dict(text=title, font=dict(size=11, color=C["muted"])),
        showgrid=grid, gridcolor="#edeae5", zeroline=False,
        tickfont=dict(color=C["muted"], size=10),
        ticksuffix=suffix,
        showline=True, linecolor=C["grey"], linewidth=1,
        **kw,
    )

# ── HTML builders ─────────────────────────────────────────────────────────────
def md(html: str):
    st.markdown(html, unsafe_allow_html=True)

def section(label: str):
    md(f"<div class='section-label'>{label}</div>")

def page(label: str):
    md(f"<div class='page-title'>{label}</div>")

def metric_card(label: str, value: str, sub: str = "", color: str = "") -> str:
    val_style = f" style='color:{color};'" if color else ""
    sub_html  = f"<div class='metric-sub'>{sub}</div>" if sub else ""
    return f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'{val_style}>{value}</div>{sub_html}</div>"

def info_card(label: str, body: str) -> str:
    return f"""
    <div class='metric-card'>
      <div class='metric-label'>{label}</div>
      <div style='font-size:0.85rem;color:#7a7974;line-height:1.6;margin-top:6px;'>{body}</div>
    </div><br>"""

def breakeven_table_html(matrix, lfpr_labels, ur_labels, hi_row, hi_col, v_max, v_min) -> str:
    hl = "color:#01696f;font-weight:700;"

    def th(label, hi=False):
        return f"<th{' style=' + repr(hl) if hi else ''}>{label}</th>"

    def td_cell(val, is_hi):
        bg, fg = cell_colours(val, v_min, v_max)
        cls = " class='cell-intersection'" if is_hi else ""
        return f"<td{cls} style='background:{bg};color:{fg};'>{signed(val)}{val:,}</td>"

    header = "<th class='row-header'>PR / UR</th>" + "".join(
        th(ur, j == hi_col) for j, ur in enumerate(ur_labels)
    )
    rows = "".join(
        f"<tr><td class='row-label'{' style=' + repr(hl) if i == hi_row else ''}>{lfpr}</td>"
        + "".join(td_cell(v, i == hi_row and j == hi_col) for j, v in enumerate(row))
        + "</tr>"
        for i, (lfpr, row) in enumerate(zip(lfpr_labels, matrix))
    )
    return f"<div style='overflow-x:auto;border:1px solid #dcd9d5;border-radius:8px;'><table class='breakeven-table'><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>"

def layoffs_table_html(df: pd.DataFrame) -> str:
    TH = ("background:#f0ede8;color:#7a7974;font-weight:600;font-size:0.78rem;"
          "text-transform:uppercase;letter-spacing:0.03em;padding:9px 14px;"
          "border-bottom:1px solid #dcd9d5;text-align:left;white-space:nowrap;"
          "position:sticky;top:0;z-index:1;")
    TD = "padding:8px 14px;border-bottom:1px solid #edeae5;vertical-align:top;"

    COLS = {
        "company":                   ("Company",  "font-weight:600;color:#28251d;white-space:nowrap;"),
        "date_of_announcement":      ("Date Announced",     "color:#7a7974;white-space:nowrap;"),
        "number_of_layoffs_planned": ("Number of Layoffs",  "text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;"),
        "sector":                    ("Sector",   "white-space:nowrap;"),
        "one_line_comment":          ("Comment",  "color:#7a7974;font-size:0.82rem;"),
    }

    header = "".join(f"<th style='{TH}'>{lbl}</th>" for lbl, _ in COLS.values())

    df = df.copy()
    df["_d"] = pd.to_datetime(df["date_of_announcement"], errors="coerce")
    df = df.sort_values("_d", ascending=False).drop(columns="_d")

    def fmt_date(v):
        try:    return pd.to_datetime(v).strftime("%b %Y")
        except: return str(v)

    def fmt_layoffs(v):
        try:    return f"{int(v):,}"
        except: return str(v)

    def pill(text):
        return f"<span style='background:#f0ede8;color:#7a7974;font-size:0.72rem;padding:2px 8px;border-radius:4px;'>{text}</span>"

    rows = ""
    for i, (_, r) in enumerate(df.iterrows()):
        bg    = "#ffffff" if i % 2 == 0 else "#f9f8f5"
        vals  = [r["company"], fmt_date(r["date_of_announcement"]),
                 fmt_layoffs(r["number_of_layoffs_planned"]), pill(r["sector"]), r["one_line_comment"]]
        cells = "".join(
            f"<td style='background:{bg};{TD}{sty}'>{v}</td>"
            for v, (_, sty) in zip(vals, COLS.values())
        )
        rows += f"<tr class='layoffs-row'>{cells}</tr>"

    return (
        f"<div style='overflow-y:auto;max-height:570px;border:1px solid #dcd9d5;border-radius:8px;'>"
        f"<table style='width:100%;border-collapse:collapse;font-size:0.875rem;font-family:Inter,sans-serif;'>"
        f"<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>"
        f"<p style='font-size:0.7rem;color:#bab9b4;margin-top:0.5rem;'>"
        f"{len(df):,} announcements in the last twelve months &nbsp;·&nbsp; collected by ChatGPT API &nbsp;·&nbsp; scroll to view all</p>"
    )

# ── Charts ────────────────────────────────────────────────────────────────────

def trend_chart(df: pd.DataFrame, breakeven_nfp: float) -> go.Figure:
    be_color = value_color(breakeven_nfp, "#437a22", neg="#a12c7b")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["nfp"], fill="tozeroy", fillcolor="rgba(1,105,111,0.08)",
        line=dict(color=COLORS["teal"], width=2), name="NFP EMA",
        hovertemplate="%{x|%b %Y}<br>Trend NFP Δ: <b>%{y:.0f}k</b><extra></extra>",
    ))
    fig.add_hline(y=breakeven_nfp, line=dict(color=be_color, width=1.5, dash="dash"),
                  annotation_text=f"Breakeven: {signed(breakeven_nfp)}{breakeven_nfp:,}k",
                  annotation_position="top right",
                  annotation_font=dict(color=be_color, size=11))
    fig.add_hline(y=0, line=dict(color=COLORS["grey"], width=1))
    fig.update_layout(**base_layout(340, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=COLORS["muted"])),
        xaxis=dict(tickformat="%b %Y"),
        yaxis=dict(ticksuffix="k"),
    ))
    return fig


def beveridge_chart(df: pd.DataFrame, include_covid: bool) -> go.Figure:
    covid_mask  = (df.index >= COVID_START) & (df.index <= COVID_END)
    pre_post    = df[~covid_mask].copy()
    covid       = df[covid_mask].copy()
    n           = len(pre_post)
    NAVY, AMBER, GOLD = (30, 60, 114), (217, 119, 6), (250, 204, 21)

    def pos_color(i):
        t = i / max(n - 1, 1)
        return lerp(NAVY, AMBER, t * 2) if t < 0.5 else lerp(AMBER, GOLD, (t - 0.5) * 2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pre_post["unemployment rate"], y=pre_post["job openings"],
        mode="markers", name="", showlegend=False,
        marker=dict(color=[pos_color(i) for i in range(n)], size=8, opacity=0.85,
                    line=dict(color="rgba(255,255,255,0.4)", width=0.5)),
        text=[d.strftime("%b %Y") for d in pre_post.index],
        hovertemplate="<b>%{text}</b><br>Unemployment: <b>%{x:.1f}%</b><br>Job Openings: <b>%{y:.1f}%</b><extra></extra>",
    ))
    # Legend anchors for gradient endpoints
    for label, rgb in [("2001", NAVY), ("2026", GOLD)]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                 marker=dict(color=f"rgb{rgb}", size=8), showlegend=True))
    if include_covid and not covid.empty:
        fig.add_trace(go.Scatter(
            x=covid["unemployment rate"], y=covid["job openings"],
            mode="markers", name="Covid (2020–21)",
            marker=dict(color="#c0bdb8", size=6, opacity=0.6,
                        line=dict(color="rgba(255,255,255,0.4)", width=0.5)),
            text=[d.strftime("%b %Y") for d in covid.index],
            hovertemplate="<b>%{text}</b><br>Unemployment: <b>%{x:.1f}%</b><br>Job Openings: <b>%{y:.1f}%</b><extra></extra>",
        ))
    latest = pre_post.iloc[-1]
    latest_date = pre_post.index[-1].strftime("%b %Y")
    fig.add_trace(go.Scatter(
        x=[latest["unemployment rate"]], y=[latest["job openings"]],
        mode="markers+text", name="Latest",
        marker=dict(color="#f43f5e", size=13, line=dict(color="#ffffff", width=2.5)),
        text=[f"  {latest_date}"], textposition="middle right",
        textfont=dict(color="#f43f5e", size=10, family="Inter"),
        customdata=[latest_date],
        hovertemplate="<b>Latest · %{customdata}</b><br>Unemployment: <b>%{x:.1f}%</b><br>Job Openings: <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig.update_layout(**base_layout(460, hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor=COLORS["grey"], borderwidth=1,
                    font=dict(size=10, color=COLORS["muted"])),
        xaxis=axis("Unemployment Rate (%)", suffix="%"),
        yaxis=axis("Job Openings Rate (%)",  suffix="%"),
    ))
    return fig


def sector_vu_chart(df: pd.DataFrame) -> go.Figure:
    latest    = df.iloc[-1]
    median    = df.median()
    order     = latest.sort_values(ascending=True).index.tolist()
    sectors   = [SECTOR_LABELS.get(s, s) for s in order]
    lat_val   = [latest[s] for s in order]
    med_val   = [median[s] for s in order]
    bar_colors = [COLORS["teal"] if lv >= mv else "#cc4125" for lv, mv in zip(lat_val, med_val)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lat_val, y=sectors, orientation="h", name="Latest",
        marker=dict(color=bar_colors, opacity=0.85),
        hovertemplate="<b>%{y}</b><br>Latest: <b>%{x:.2f}</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=med_val, y=sectors, mode="markers", name="10-Year Median",
        marker=dict(symbol="diamond", color=COLORS["text"], size=8, line=dict(color="#ffffff", width=1)),
        hovertemplate="<b>%{y}</b><br>Median: <b>%{x:.2f}</b><extra></extra>",
    ))
    fig.update_layout(**base_layout(500, hovermode="y unified", bargap=0.35,
        margin=dict(l=0, r=16, t=16, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=COLORS["muted"])),
        xaxis=axis("Job Openings / Unemployed"),
        yaxis=dict(showgrid=False, zeroline=False, showline=False,
                   tickfont=dict(color=COLORS["muted"], size=10)),
    ))
    return fig

# ── App ───────────────────────────────────────────────────────────────────────

PLOTLY_CFG = {"displayModeBar": False}

def main():
    st.set_page_config(page_title="US Labor Market Dynamics", page_icon="📊", layout="wide")
    md(CSS)

    md("<div class='app-main-title'>US Labor Market Dynamics</div>")
    page("Job Growth Breakeven Rates vs. Trend Growth")

    # Controls
    col_m, col_n, *_ = st.columns([1, 1, 4])
    with col_m:
        net_migration = st.number_input("Net Immigration Growth Est. (%)", value=0.44, step=0.02, format="%.2f")
    with col_n:
        months = st.selectbox("Horizon (months)", options=[1, 3, 6, 12], index=3)
    md("<p style='font-size:0.75rem;color:#7a7974;font-style:italic;margin-bottom:0.5rem;'>"
       "*2026 immigration estimates by source ·&nbsp; Brookings: -0.26%&nbsp;·&nbsp;CBO: +0.44%&nbsp;·&nbsp;Census: +0.25%"
       "</p><hr style='border-color:#dcd9d5;'/>")

    # Data
    df_trend     = compute_trend()
    trend_latest = df_trend["nfp"].iloc[-1]
    df_matrix, last_lfpr, last_ur, last_date, exp_ur = compute_matrix(months, net_migration, trend_latest)
    sector_vu    = compute_sectors()
    layoffs      = load_layoffs()

    lfpr_labels, ur_labels = list(df_matrix.index), list(df_matrix.columns)
    matrix   = df_matrix.values.tolist()
    all_vals = [v for row in matrix for v in row]
    v_max    = max((v for v in all_vals if v > 0), default=1)
    v_min    = min((v for v in all_vals if v < 0), default=-1)

    nearest = lambda labels, val: min(range(len(labels)), key=lambda i: abs(float(labels[i].strip("%")) - val))
    lfpr_idx, ur_idx  = nearest(lfpr_labels, last_lfpr), nearest(ur_labels, last_ur)
    breakeven_nfp     = matrix[lfpr_idx][ur_idx]

    # Metric cards
    for col, kwargs in zip(st.columns(5), [
        dict(
            label="Unemployment Rate (UR)",
            value=f"{last_ur:.2f}%",
            sub=f"as of {dt.date.today()}"
        ),
        dict(
            label="Participation Rate (PR)",
            value=f"{last_lfpr:.2f}%",
            sub=f"as of {dt.date.today()}"
        ),
        dict(
            label="Breakeven NFP Rate",
            value=f"{signed(breakeven_nfp)}{breakeven_nfp:,}k",
            sub=f"at {lfpr_labels[lfpr_idx]} PR · {ur_labels[ur_idx]} UR",
            color=value_color(breakeven_nfp, "#32CD32")
        ),
        dict(
            label="Trend NFP Rate",
            value=f"{signed(trend_latest)}{trend_latest:,.0f}k",
            sub=f"as of {dt.date.today()}",
            color=value_color(trend_latest, "#32CD32")
        ),
        dict(
            label="Expected Unemployment Rate",
            value=f"{exp_ur:.2f}%",
            sub=f"at {lfpr_labels[lfpr_idx]} PR · {signed(trend_latest)}{trend_latest:,.0f}k trend",
            color="#899499"
        ),
    ]):
        col.markdown(metric_card(**kwargs), unsafe_allow_html=True)

    md("<br>")
    md(info_card("Interpretation & Commentary",
        f"The breakeven matrix shows the average monthly payroll growth over a <strong style='color:#28251d;'>{months}-month</strong> "
        f"horizon needed to maintain the current unemployment rate, conditional on a specified participation rate and population growth path."
        f" With trend payrolls running near <strong style='color:#28251d;'>{signed(trend_latest)}{trend_latest:,.0f}k </strong>"
        f" versus a <strong style='color:#28251d;'> {signed(breakeven_nfp)}{breakeven_nfp:,}k </strong> "
        f"breakeven pace, current hiring would shift the unemployment rate towards roughly <strong style='color:#28251d;'> {exp_ur:.2f}% </strong> "
        f"over the horizon period")
       )

    # Breakeven table
    section("Non-farm Payrolls Breakeven Matrix (thousands)")
    md(breakeven_table_html(matrix, lfpr_labels, ur_labels, lfpr_idx, ur_idx, v_max, v_min))

    # Trend chart
    md("<br>")
    section("Trend Growth · Exponential Moving Average of Non-farm Payroll Changes · Last Five Years")
    st.plotly_chart(trend_chart(df_trend, breakeven_nfp), use_container_width=True, config=PLOTLY_CFG)
    md("<hr>")

    # Beveridge curve & sector slack
    page("Beveridge Curve & Sector Slack Indicators")
    md(info_card("Interpretation & Commentary",
        "US labor demand is softening as job openings continue to trend lower, moving the economy along the "
        "Beveridge curve into a zone where further vacancy declines are more likely to translate into a sharper "
        "rise in unemployment. Across most major sectors, openings per unemployed worker have fallen below their "
        "10‑year medians, with particularly pronounced slack in government and professional services, consistent"
        "with reduced hiring linked to DOGE restructuring and early AI‑related displacement."))

    col_left, col_right = st.columns(2)
    with col_left:
        col_lbl, col_tog = st.columns([2, 1])
        with col_lbl:
            section("Job Openings Rate vs. Unemployment Rate")
        with col_tog:
            st.markdown('<div class="bev-toggle">', unsafe_allow_html=True)
            include_covid = st.toggle("include Covid (2020–21)", value=False)
            st.markdown('</div>', unsafe_allow_html=True)
        st.plotly_chart(beveridge_chart(compute_beveridge(include_covid), include_covid),
                        use_container_width=True, config=PLOTLY_CFG)
    with col_right:
        section("Ratio of Job Openings to Unemployment by Sector")
        st.plotly_chart(sector_vu_chart(sector_vu), use_container_width=True, config=PLOTLY_CFG)
    md("<hr>")

    # Layoffs monitor
    page("Layoffs Monitor")
    section("Recent Corporate Layoff Announcements")
    md(layoffs_table_html(layoffs))


if __name__ == "__main__":
    main()