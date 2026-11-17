"""
plot_cooking_from_SAS.py
========================
Cooking behaviour of soaked WHOLE cowpea seeds – figures made directly from
your SAS output.

Input files (keep them in the same folder as this script):
  1. RELATIVE_cooking_kinetics_output_soaked_cowpeas_oct_2026_1.xlsx
       SAS PROC NLIN output: one sheet per variety with the output table
       Obs | Time | RH | predRH | k | RHf   (RH of every individual seed)
       + sheet "Summary 6 var" with k, SE k, RHf, SE RHf
  2. soaked_and_unsoaked_current_analysis-logistic_2.xlsx
       SAS PROC LOGISTIC output: sheet "Soaked cooking time" with
       var | Ctime | Ntotal | Ncooked | Cookpercent | predicted | lowlimit | uplimit
       + sheet "CT95" with CT95 and its StdErr

Output (folder "Whole seed figures from SAS"):
  Fig A - cooking curves (relative hardness).png
  Fig B - CT95 curves (logistic).png
  Fig C - CT95 bars.png
  Legend - varieties.png          (stand-alone legend, e.g. for slides)
  SAS parameters used.csv         (the numbers behind the figures)

Tested with Python 3.13; works the same on Python 3.14 in Spyder.
Packages: numpy, pandas, matplotlib, openpyxl (all included in Anaconda).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =============================================================================
# 0. SETTINGS – change file names, colours or order here (file names can be changed here, when reusing the script)
# =============================================================================
SAS_NLIN_FILE = "RELATIVE_cooking_kinetics_output_soaked_cowpeas_oct_2026_1.xlsx"
SAS_LOGISTIC_FILE = "soaked_and_unsoaked_current_analysis-logistic_2.xlsx"
OUT = Path("Whole seed figures from SAS")
OUT.mkdir(exist_ok=True)

# Varieties in order of CT95 (shortest -> longest); this order is used everywhere
VARIETIES = ["Kat Kunde", "Kunde Soko", "Katumani 80", "Kunde Tamu", "Machakos 66", "KVU-27-1"]

# Name of each variety's sheet in the SAS NLIN file
NLIN_SHEET = {"Kat Kunde": "Kat Kunde", "Kunde Soko": "Kunde Soko", "Katumani 80": "K80",
              "Kunde Tamu": "Kunde Tamu", "Machakos 66": "Machakos66", "KVU-27-1": "KVU27"}

COLOUR = dict(zip(VARIETIES, ["#1a4f8b", "#a8241c", "#1d6b37", "#b85000", "#5e2a84", "#b0306e"]))
MARKER = dict(zip(VARIETIES, ["o", "s", "^", "D", "v", "P"]))
LINE = {"Kat Kunde": (0, (6, 2)), "Kunde Soko": "-", "Katumani 80": "-",
        "Kunde Tamu": (0, (1.5, 1.5)), "Machakos 66": "-", "KVU-27-1": (0, (6, 2))}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 14, "axes.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelweight": "bold", "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white", bbox_inches="tight")
    plt.show()          # shows the figure in Spyder's Plots pane
    plt.close(fig)
    print("saved:", name)


# =============================================================================
# 1. READ THE SAS NLIN OUTPUT (relative hardness of every seed + k, RHf)
# =============================================================================
def read_sas_output_table(ws):
    """Find the SAS output table (header Obs, Time, RH, predRH, k, RHf) anywhere
    on the sheet and return it as a DataFrame. The table is not always in the
    same columns, so we search for the header row."""
    rows = list(ws.iter_rows(values_only=True))
    for r, row in enumerate(rows):
        row = list(row)
        for c in range(len(row) - 5):
            if row[c] == "Obs" and row[c + 1] == "Time" and row[c + 2] == "RH" and row[c + 3] == "predRH":
                data = [list(x[c:c + 6]) for x in rows[r + 1:]]
                df = pd.DataFrame(data, columns=["Obs", "Time", "RH", "predRH", "k", "RHf"])
                return df.apply(pd.to_numeric, errors="coerce").dropna(subset=["Time", "RH"])
    raise ValueError(f"No SAS output table found on sheet {ws.title}")


wb = openpyxl.load_workbook(SAS_NLIN_FILE, read_only=True, data_only=True)
RH = {v: read_sas_output_table(wb[NLIN_SHEET[v]]) for v in VARIETIES}

# k, SE k, RHf, SE RHf from the SAS summary sheet
summ = pd.read_excel(SAS_NLIN_FILE, sheet_name="Summary 6 var", header=1, usecols="B:F", nrows=6)
summ.columns = ["variety", "k", "SE_k", "RHf", "SE_RHf"]
summ = summ.set_index("variety")

# =============================================================================
# 2. READ THE SAS LOGISTIC OUTPUT (% cooked seeds, predicted, CT95 and SE)
# =============================================================================
logi = pd.read_excel(SAS_LOGISTIC_FILE, sheet_name="Soaked cooking time")
ct = pd.read_excel(SAS_LOGISTIC_FILE, sheet_name="CT95", header=None, skiprows=21, usecols="B:D", nrows=6)
ct.columns = ["variety", "CT95", "SE_CT95"]
ct = ct.set_index("variety")

# The logistic curve is logit(p) = a + b*t. SAS gives the predicted p at each
# time point, so a and b follow exactly from a straight line through logit(p).
AB = {}
for v in VARIETIES:
    g = logi[logi["var"] == v]
    p = g["predicted"].clip(1e-9, 1 - 1e-9)
    b, a = np.polyfit(g["Ctime"], np.log(p / (1 - p)), 1)
    AB[v] = (a, b)

# =============================================================================
# 3. TABLE OF THE SAS PARAMETERS USED
# =============================================================================
rows = []
for v in VARIETIES:
    d = RH[v]
    sse = ((d["RH"] - d["predRH"]) ** 2).sum()
    r2 = 1 - sse / ((d["RH"] - d["RH"].mean()) ** 2).sum()
    rows.append({"Variety": v, "n seeds": len(d),
                 "k (1/min)": summ.loc[v, "k"], "SE k": summ.loc[v, "SE_k"],
                 "RHf": summ.loc[v, "RHf"], "SE RHf": summ.loc[v, "SE_RHf"], "R2": round(r2, 3),
                 "a": round(AB[v][0], 4), "b (1/min)": round(AB[v][1], 5),
                 "CT95 (min)": ct.loc[v, "CT95"], "SE CT95 (min)": ct.loc[v, "SE_CT95"]})
params = pd.DataFrame(rows)
params.to_csv(OUT / "SAS parameters used.csv", index=False, encoding="utf-8-sig")
print(params.to_string(index=False))


def legend_handles(with_marker=True):
    return [Line2D([0], [0], color=COLOUR[v], lw=2.6, ls=LINE[v],
                   marker=MARKER[v] if with_marker else None, mfc="white", mew=1.6, ms=7)
            for v in VARIETIES]


# =============================================================================
# 4. FIG A – COOKING CURVES: relative hardness vs time + SAS model lines
# =============================================================================
def frac_conv(t, k, RHf):
    return RHf + (1 - RHf) * np.exp(-k * t)


fig, ax = plt.subplots(figsize=(10.5, 6.6))
shift = dict(zip(VARIETIES, np.linspace(-2.5, 2.5, 6)))      # small shift so error bars do not overlap
for v in VARIETIES:
    d = RH[v]
    m = d.groupby("Time")["RH"].agg(["mean", "std"])          # mean ± SD per time point
    ax.errorbar(m.index + shift[v], m["mean"], yerr=m["std"], fmt=MARKER[v], ms=6.5,
                color=COLOUR[v], mfc="white", mew=1.6, capsize=2.5, lw=1, zorder=3)
    t = np.linspace(0, d["Time"].max(), 300)
    ax.plot(t, frac_conv(t, summ.loc[v, "k"], summ.loc[v, "RHf"]), color=COLOUR[v], lw=2.6,
            ls=LINE[v], zorder=4)
labels = [f'{v}: k = {summ.loc[v, "k"]:.4f} min$^{{-1}}$, RH$_f$ = {summ.loc[v, "RHf"]:.2f}' for v in VARIETIES]
leg = ax.legend(legend_handles(), labels, frameon=False, fontsize=12, loc="upper right", handlelength=3.4)
for txt, v in zip(leg.get_texts(), VARIETIES):
    txt.set_color(COLOUR[v])
ax.text(0.02, 0.04, "Lines: fractional conversion model, RH(t) = RH$_f$ + (1 − RH$_f$)·exp(−k·t) (SAS NLIN)\n"
        "Points: mean ± SD (n = 20 soaked whole seeds per time point)",
        transform=ax.transAxes, fontsize=11.5, color="#444")
ax.set_xlabel("Cooking time at 95 °C (min)")
ax.set_ylabel("Relative hardness, H/H$_0$ (–)")
ax.set_xlim(-5, 172)
ax.set_ylim(0, 1.2)
save(fig, "Fig A - cooking curves (relative hardness).png")

# =============================================================================
# 5. FIG B – CT95 CURVES: % cooked seeds + SAS logistic model
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 6.4))
t = np.linspace(0, 170, 400)
ax.axhline(95, color="#888", ls=":", lw=1.3)
ax.text(171, 95, "95%", va="center", fontsize=12, color="#666")
for v in VARIETIES:
    g = logi[logi["var"] == v]
    a, b = AB[v]
    ax.plot(t, 100 / (1 + np.exp(-(a + b * t))), color=COLOUR[v], lw=2.6, ls=LINE[v])
    ax.scatter(g["Ctime"], g["Cookpercent"], marker=MARKER[v], s=55, color=COLOUR[v],
               edgecolor="white", lw=0.8, zorder=3)
    ax.plot([ct.loc[v, "CT95"]] * 2, [0, 95], color=COLOUR[v], lw=1.2, ls=":", zorder=1)
ax.legend(legend_handles(), [f'{v}  ({ct.loc[v, "CT95"]:.0f} min)' for v in VARIETIES],
          title="Variety (CT$_{95}$)", title_fontsize=14, frameon=False, fontsize=13.5,
          loc="center left", bbox_to_anchor=(1.04, 0.5), handlelength=3.2)
ax.text(1.05, 0.12, "Points: % cooked seeds\n(n = 20 per time point)\nLines: logistic model (SAS)\nDotted: CT$_{95}$",
        transform=ax.transAxes, va="top", fontsize=12.5, color="#555")
ax.set_xlabel("Cooking time at 95 °C (min)")
ax.set_ylabel("Cooked seeds (%)")
ax.set_xlim(0, 175)
ax.set_ylim(-2, 104)
save(fig, "Fig B - CT95 curves (logistic).png")

# =============================================================================
# 6. FIG C – CT95 BARS ± SAS standard error
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5.8))
for i, v in enumerate(VARIETIES):
    m, se = ct.loc[v, "CT95"], ct.loc[v, "SE_CT95"]
    ax.bar(i, m, 0.65, color=COLOUR[v])
    ax.errorbar(i, m, yerr=se, color="black", capsize=6, lw=1.6)
    ax.text(i, 8, f"{m:.1f}\nmin", ha="center", fontsize=13.5, color="white", fontweight="bold")
ax.set_xticks(range(len(VARIETIES)))
ax.set_xticklabels(VARIETIES, rotation=20, ha="right")
for txt, v in zip(ax.get_xticklabels(), VARIETIES):
    txt.set_color(COLOUR[v])
    txt.set_fontweight("bold")
ax.set_ylabel("CT$_{95}$ (min)")
ax.set_ylim(0, 135)
ax.text(0.01, 0.98, "Bars: CT$_{95}$ from the Cooking logistic modelling; error bars: SE (SAS)",
        transform=ax.transAxes, va="top", fontsize=11.5, color="#444")
save(fig, "Fig C - CT95 bars.png")

# =============================================================================
# 7. STAND-ALONE LEGEND (useful when you place figures side by side)
# =============================================================================
fig = plt.figure(figsize=(9, 1.0))
fig.legend(legend_handles(), VARIETIES, loc="center", ncol=6, frameon=False, fontsize=13, handlelength=3)
save(fig, "Legend - varieties.png")

print(f"\nAll figures are in: {OUT.resolve()}")
