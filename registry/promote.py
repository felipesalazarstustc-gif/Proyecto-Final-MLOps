import json, os, time
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/"evidencias"/"promotion_log.jsonl"
MODEL=os.getenv("MLFLOW_MODEL_NAME","OrvexaRegressor")
TRACK=os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000")

def get_baseline():
    p=ROOT/"data/artifacts/baseline.json"
    if not p.exists(): raise FileNotFoundError("Run baseline first: python registry/promote.py --baseline")
    return json.loads(p.read_text())

def promote(candidate_version):
    mlflow.set_tracking_uri(TRACK); c=MlflowClient()
    base=get_baseline()
    v=c.get_model_version(MODEL,str(candidate_version))
    run=c.get_run(v.run_id)
    rmse=float(run.data.metrics["rmse_val"])
    ms=float(run.data.metrics["inference_ms_per_observation"])
    ok_rmse=rmse < 0.95*base["rmse"]
    ok_time=ms <= 2.0*base["ms_obs"]
    decision="PROMOTE" if ok_rmse and ok_time else "STAGING"
    reason=[]
    if not ok_rmse: reason.append("RMSE no es al menos 5% mejor que la línea base")
    if not ok_time: reason.append("tiempo de inferencia supera 2x la línea base")
    if decision=="PROMOTE":
        for old in c.get_latest_versions(MODEL,stages=["Production"]):
            if old.version != str(candidate_version):
                c.transition_model_version_stage(MODEL,old.version,stage="Archived",archive_existing_versions=False)
        c.transition_model_version_stage(MODEL,str(candidate_version),stage="Production",archive_existing_versions=True)
        tag="promoted"
    else:
        c.transition_model_version_stage(MODEL,str(candidate_version),stage="Staging",archive_existing_versions=False)
        tag="failed:" + "; ".join(reason)
    c.set_model_version_tag(MODEL,str(candidate_version),"promotion_status",tag)
    rec={"timestamp":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
         "version":str(candidate_version),"decision":decision,"reason":reason,
         "rmse":rmse,"baseline_rmse":base["rmse"],"ms_obs":ms,"baseline_ms_obs":base["ms_obs"]}
    LOG.parent.mkdir(exist_ok=True); LOG.open("a",encoding="utf-8").write(json.dumps(rec)+"\n")
    print(json.dumps(rec,indent=2))

def save_baseline():
    mlflow.set_tracking_uri(TRACK); c=MlflowClient()
    # Baseline is the first run tagged baseline=true.
    runs=c.search_runs(filter_string="tags.baseline = 'true'",max_results=1,order_by=["start_time ASC"])
    if not runs: raise RuntimeError("No baseline run found")
    r=runs[0]
    data={"run_id":r.info.run_id,"rmse":r.data.metrics["rmse_val"],
          "ms_obs":r.data.metrics["inference_ms_per_observation"]}
    (ROOT/"data/artifacts/baseline.json").write_text(json.dumps(data,indent=2))
    print(json.dumps(data,indent=2))

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--version",type=int)
    ap.add_argument("--baseline",action="store_true")
    a=ap.parse_args()
    if a.baseline: save_baseline()
    elif a.version: promote(a.version)
    else: ap.print_help()
