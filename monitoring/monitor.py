import json, os, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
BATCH_DIR=ROOT/"data/batches"
REPORT=ROOT/"evidencias/drift_report.json"
ALERT=ROOT/"evidencias/alerts.log"
NUMERIC=["Hour","TemperatureC","Humidity","Wind_speed_ms","Visibility10m",
         "Dew_point_temperatureC","Solar_RadiationMJm2","Rainfallmm","Snowfall_cm",
         "year","month_num","day","weekday_num"]
CATEGORICAL=["Seasons","Holiday","Functioning_Day"]
ALPHA=0.05

def monitor_batch(batch):
    ref=pd.read_csv(BATCH_DIR/"L0.csv"); cur=pd.read_csv(BATCH_DIR/f"{batch}.csv")
    rows=[]
    pvals=[]; keys=[]
    for col in NUMERIC:
        stat,p=ks_2samp(ref[col].dropna(),cur[col].dropna())
        rows.append({"batch":batch,"variable":col,"test":"KS","statistic":float(stat),"p_raw":float(p)})
        pvals.append(p); keys.append(len(rows)-1)
    for col in CATEGORICAL:
        cats=sorted(set(ref[col].dropna().astype(str))|set(cur[col].dropna().astype(str)))
        a=np.array([[(ref[col].astype(str)==x).sum() for x in cats],
                    [(cur[col].astype(str)==x).sum() for x in cats]])
        stat,p,_,_=chi2_contingency(a)
        rows.append({"batch":batch,"variable":col,"test":"chi2","statistic":float(stat),"p_raw":float(p)})
        pvals.append(p); keys.append(len(rows)-1)
    reject,p_adj,_,_=multipletests(pvals,alpha=ALPHA,method="fdr_bh")
    alerts=[]
    for i,r in enumerate(rows):
        r["p_adjusted"]=float(p_adj[i]); r["drift"]=bool(reject[i])
        if r["drift"]: alerts.append(r)
    data=json.loads(REPORT.read_text()) if REPORT.exists() else []
    data.append({"batch":batch,"variables":rows})
    REPORT.write_text(json.dumps(data,indent=2))
    if alerts:
        with ALERT.open("a",encoding="utf-8") as f:
            for a in alerts:
                f.write(json.dumps({"batch":batch,"variable":a["variable"],"p_adjusted":a["p_adjusted"],
                                    "statistic":a["statistic"]})+"\n")
    return {"alerts":alerts,"total_tested":len(rows)}

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--batch",required=True)
    print(json.dumps(monitor_batch(ap.parse_args().batch),indent=2))
