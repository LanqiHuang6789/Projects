"""
Final Project — BIS634
Author: Lanqi Huang
Topic: Structural Food Environments and Health Disparities in the U.S.
"""

from __future__ import annotations
import os, json
from flask import Flask, render_template, request, jsonify, send_file, render_template_string
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = os.path.dirname(__file__)
DATA_CSV = os.path.join(BASE_DIR, "data", "StateAndCountyData.csv")
VARLIST_PATH = os.path.join(BASE_DIR, "data", "VariableList.csv")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

df = pd.read_csv(DATA_CSV)
varmeta = pd.read_csv(VARLIST_PATH) if os.path.exists(VARLIST_PATH) else pd.DataFrame()


def safe_mean(sub: pd.DataFrame, var: str) -> float:
    vals = sub[sub["Variable_Code"] == var]["Value"]
    vals = vals[(vals.notna()) & (vals > 0)]
    return float(vals.mean()) if not vals.empty else float("nan")

def state_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["Value"].notna() & (df["Value"] > 0)]
    wide = sub.pivot_table(index="State", columns="Variable_Code", values="Value", aggfunc="mean")
    wide.reset_index(inplace=True)
    return wide

wide_df = state_feature_matrix(df)


state_abbrev = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","District of Columbia":"DC",
    "Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL",
    "Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
    "Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN",
    "Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
    "New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY",
    "North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR",
    "Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
    "Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA",
    "Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY"
}
inv_state_name = {v:k for k,v in state_abbrev.items()}

FOCUS_VARS = [
    # Food access (2015–2019)
    "PCT_LACCESS_POP15", "PCT_LACCESS_POP19",

    # Obesity & Diabetes (2015–2022)
    "PCT_DIABETES_ADULTS15", "PCT_DIABETES_ADULTS19",
    "PCT_OBESE_ADULTS17", "PCT_OBESE_ADULTS22",

    # Poverty & Income (2021)
    "POVRATE21", "MEDHHINC21",

    # Grocery stores (2016–2020)
    "GROC16", "GROC20",

    # Metro classification (rural/urban proxy)
    "METRO23",

    # Assistance and insecurity (2021–2023)
    "PCT_SNAP22",            # SNAP participation (% pop)
    "PC_WIC_REDEMP22",       # WIC redemptions per capita
    "FOODINSEC_21_23",       # Household food insecurity (%)
    "VLFOODSEC_21_23"        # Very low food security (%)
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    state = request.form["usertext"].title()
    abbr = state_abbrev.get(state, None)
    if not abbr:
        return render_template("analyze.html", state=state, result="State not found.", chart_url=None)

    row = df[df["State"] == abbr]
    if row.empty:
        return render_template("analyze.html", state=state, result=f"No data for {state}.", chart_url=None)

    food_insec = safe_mean(row, "FOODINSEC_21_23")
    obesity = safe_mean(row, "PCT_OBESE_ADULTS22")
    poverty = safe_mean(row, "POVRATE21")
    income = safe_mean(row, "MEDHHINC21")

    avg_food_insec = safe_mean(df, "FOODINSEC_21_23")
    avg_obesity = safe_mean(df, "PCT_OBESE_ADULTS22")
    avg_poverty = safe_mean(df, "POVRATE21")
    avg_income = safe_mean(df, "MEDHHINC21")

    result_text = (
        f"In {state} ({abbr}), {food_insec:.1f}% of households are food insecure "
        f"and {obesity:.1f}% of adults are obese. "
        f"Poverty rate is {poverty:.1f}% (vs {avg_poverty:.1f}% nationally), "
        f"median income ${income:,.0f} (vs ${avg_income:,.0f} U.S. average)."
    )

    plt.figure(figsize=(4,3))
    categories = ["Food Insecurity %","Obesity %","Poverty %"]
    state_vals = [food_insec, obesity, poverty]
    nat_vals = [avg_food_insec, avg_obesity, avg_poverty]
    x = range(len(categories))
    plt.bar(x, state_vals, width=0.4, label=state, color="#4B9CD3")
    plt.bar([i+0.4 for i in x], nat_vals, width=0.4, label="National", color="#F39C12")
    plt.xticks([i+0.2 for i in x], categories)
    plt.ylabel("Percentage")
    plt.title(f"{state} vs National Averages")
    plt.legend()
    plt.tight_layout()
    chart_path = os.path.join("static", "chart.png")
    plt.savefig(chart_path)
    plt.close()

    return render_template("analyze.html", state=state, result=result_text, chart_url="chart.png")

@app.route("/dashboard")
def dashboard():
    html = open(os.path.join(app.template_folder, "dashboard.html")).read()
    return render_template_string(html)

@app.route("/api/variables")
def api_variables():
    info = []
    for c in FOCUS_VARS:
        desc = varmeta.loc[varmeta["Variable_Code"] == c, "Variable_Name"]
        info.append({
            "Variable_Code": c,
            "Variable_Name": desc.iloc[0] if not desc.empty else c
        })
    return jsonify({
        "rowcount": int(len(wide_df)),
        "states": wide_df["State"].tolist(),
        "variables": info
    })

@app.route("/api/summary")
def api_summary():
    numeric = wide_df[[v for v in FOCUS_VARS if v in wide_df.columns]]
    summary = numeric.describe().T.reset_index().rename(columns={"index": "Variable_Code"})
    if not varmeta.empty:
        summary = summary.merge(varmeta[["Variable_Code","Variable_Name"]], on="Variable_Code", how="left")
    return jsonify(json.loads(summary.to_json(orient="records")))

@app.route("/api/plot/corr")
def api_plot_corr():
    existing = [v for v in FOCUS_VARS if v in wide_df.columns]
    num = wide_df[existing].dropna()
    corr = num.corr(numeric_only=True)
    label_map = dict(zip(varmeta["Variable_Code"], varmeta["Variable_Name"]))
    labels = [label_map.get(v,v) for v in corr.columns]
    fig = plt.figure(figsize=(7,5), dpi=150)
    im = plt.imshow(corr, interpolation="nearest", cmap="viridis")
    plt.colorbar(im)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.title("Correlation: Food Environment, Assistance, and Health (2015–2023)")
    plt.tight_layout()
    out_path = os.path.join(STATIC_DIR, "corr.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return send_file(out_path, mimetype="image/png")

@app.route("/api/plot/scatter/<x>/<y>")
def api_plot_scatter(x, y):
    try:
        x = x.strip().upper(); y = y.strip().upper()
        cols = [c.upper() for c in wide_df.columns]
        missing = [v for v in [x,y] if v not in cols]
        if missing:
            return jsonify({"error": f"Variable(s) not found: {', '.join(missing)}"}), 200
        real_x = [c for c in wide_df.columns if c.upper()==x][0]
        real_y = [c for c in wide_df.columns if c.upper()==y][0]
        sub = wide_df[[real_x, real_y, "State"]].dropna()
        if sub.empty:
            return jsonify({"error": "No valid data"}), 400
        name_x = varmeta.loc[varmeta["Variable_Code"]==real_x, "Variable_Name"]
        name_y = varmeta.loc[varmeta["Variable_Code"]==real_y, "Variable_Name"]
        xlabel = name_x.iloc[0] if not name_x.empty else real_x
        ylabel = name_y.iloc[0] if not name_y.empty else real_y
        fig = plt.figure(figsize=(6,4), dpi=150)
        plt.scatter(sub[real_x], sub[real_y], alpha=0.7)
        for i, s in enumerate(sub["State"]):
            if i % 6 == 0:
                plt.annotate(s, (sub[real_x].iat[i], sub[real_y].iat[i]), fontsize=6, alpha=0.7)
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs {xlabel}")
        plt.tight_layout()
        out_path = os.path.join(STATIC_DIR, "scatter.png")
        fig.savefig(out_path)
        plt.close(fig)
        return send_file(out_path, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": f"Plot failed: {e}"}), 500


@app.route("/api/plot/change/<var1>/<var2>")
def api_plot_change(var1, var2):
    base = var1.strip().upper()
    follow = var2.strip().upper()
    if base not in wide_df.columns or follow not in wide_df.columns:
        return jsonify({"error": "Variables not found"}), 400
    sub = wide_df[["State", base, follow]].dropna()
    sub["Change"] = sub[follow] - sub[base]
    result = json.loads(sub[["State", "Change"]].to_json(orient="records"))
    return jsonify(result)


@app.route("/api/cluster/<int:k>")
def api_cluster(k):
    try:
        vars_list = [v for v in FOCUS_VARS if v in wide_df.columns]
        subset = wide_df[["State"] + vars_list].dropna()
        scaler = StandardScaler()
        X = scaler.fit_transform(subset[vars_list])
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        subset["cluster"] = km.fit_predict(X)
        subset["Name"] = subset["State"].map(inv_state_name)
        subset.loc[subset["State"]=="DC","Name"] = "District of Columbia"
        return jsonify(json.loads(subset[["State","Name","cluster"]].to_json(orient="records")))
    except Exception as e:
        return jsonify({"error": f"Cluster failed: {e}"}), 500


@app.route("/api/state/<state>")
def api_state(state):
    state_clean = state.strip().title()
    abbr = state_clean.upper()
    if state_clean in state_abbrev:
        abbr = state_abbrev[state_clean]
    elif abbr in inv_state_name:
        state_clean = inv_state_name[abbr]
    else:
        return jsonify({"error": f"State '{state}' not recognized."}), 404
    df["State"] = df["State"].str.upper()
    row = df[df["State"] == abbr]
    if row.empty:
        return jsonify({"error": f"No data for {state_clean} ({abbr})."}), 404
    result = {"state": state_clean, "abbr": abbr}
    for v in FOCUS_VARS:
        if v in df["Variable_Code"].unique():
            result[v] = safe_mean(row, v)
    return jsonify(result)


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", message="Page not found."), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template("error.html", message="Internal server error."), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
