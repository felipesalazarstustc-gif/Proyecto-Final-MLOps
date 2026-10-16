import json, os, sys, time, subprocess
from pathlib import Path
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/"evidencias"/"pipeline_state.json"
LOG=ROOT/"evidencias"/"pipeline_log.jsonl"
BATCH_DIR=ROOT/"data/batches"
sys.path.insert(0,str(ROOT))
from training.train import train_single
from registry.promote import promote
from monitoring.monitor import monitor_batch

EXPECTED=[
    "timestamp","Hour","TemperatureC","Humidity","Wind_speed_ms","Visibility10m",
    "Dew_point_temperatureC","Solar_RadiationMJm2","Rainfallmm","Snowfall_cm",
    "Seasons","Holiday","Functioning_Day","year","month_num","day","weekday_num","target"
]
NUM_RANGES={"Hour":(0,23),"Humidity":(0,100),"TemperatureC":(-40,60)}

def log(obj):
    LOG.parent.mkdir(exist_ok=True); LOG.open("a",encoding="utf-8").write(json.dumps(obj)+"\n")

def validate(df):
    missing=[c for c in EXPECTED if c not in df.columns]
    if missing: return False,f"faltan columnas: {missing}"
    nulls=df[EXPECTED].isna().mean()
    bad=nulls[nulls>0.05].to_dict()
    if bad: return False,f"nulos >5%: {bad}"
    for c,(lo,hi) in NUM_RANGES.items():
        if (df[c]<lo).any() or (df[c]>hi).any():
            return False,f"fuera de rango histórico: {c}"
    return True,"ok"

def next_batch():
    state=json.loads(STATE.read_text()) if STATE.exists() else {"processed":[]}
    for b in ["L1","L2","L3","L4"]:
        if b not in state["processed"]: return b,state
    return None,state

def process():
    batch,state=next_batch()
    if not batch:
        print("Todos los lotes ya fueron procesados."); return
    p=BATCH_DIR/f"{batch}.csv"
    df=pd.read_csv(p)
    ok,reason=validate(df)
    if not ok:
        rec={"batch":batch,"status":"REJECTED","reason":reason,"timestamp":time.time()}
        log(rec); state["processed"].append(batch); STATE.write_text(json.dumps(state,indent=2)); return
    result=train_single(batch,run_name=f"scheduled_{batch}")
    # Find registered version for the run.
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000"))
    c=MlflowClient(); name=os.getenv("MLFLOW_MODEL_NAME","OrvexaRegressor")
    versions=c.search_model_versions(f"name='{name}'")
    ver=max([int(v.version) for v in versions if v.run_id==result["run_id"]])
    base=json.loads((ROOT/"data/artifacts/baseline.json").read_text())
    decision_before=(result["rmse"] < 0.95*base["rmse"] and result["ms_obs"] <= 2*base["ms_obs"])
    promote(ver)
    drift=monitor_batch(batch)
    rec={"batch":batch,"status":"SUCCESS","version":ver,"decision":"PROMOTE" if decision_before else "STAGING",
         "rmse":result["rmse"],"drift":drift,"timestamp":time.time()}
    log(rec); state["processed"].append(batch); STATE.write_text(json.dumps(state,indent=2))

if __name__=="__main__":
    scheduler=BlockingScheduler()
    process()
    scheduler.add_job(process,"interval",seconds=30,id="scheduled_retraining",max_instances=1,coalesce=True)
    print("Scheduler activo: revisa un lote nuevo cada 30 segundos.")
    scheduler.start()
