import os, time, json
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODEL=os.getenv("MLFLOW_MODEL_NAME","OrvexaRegressor")
TRACK=os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000")
LOG=ROOT/"evidencias"/"rollback_log.jsonl"

mlflow.set_tracking_uri(TRACK); c=MlflowClient()
arch=c.get_latest_versions(MODEL,stages=["Archived"])
if not arch:
    print("No hay versiones Archived.")
    raise SystemExit
for v in arch:
    run=c.get_run(v.run_id)
    print(f"version={v.version} rmse={run.data.metrics.get('rmse_val')} ms={run.data.metrics.get('inference_ms_per_observation')}")

choice=input("Escriba la versión a restaurar o ENTER para cancelar: ").strip()
if not choice:
    print("Cancelado."); raise SystemExit
confirm=input(f"CONFIRME restaurar versión {choice} a Production escribiendo SI: ").strip()
if confirm!="SI":
    print("Cancelado."); raise SystemExit
for old in c.get_latest_versions(MODEL,stages=["Production"]):
    c.transition_model_version_stage(MODEL,old.version,stage="Archived",archive_existing_versions=False)
c.transition_model_version_stage(MODEL,choice,stage="Production",archive_existing_versions=False)
who=input("Quién realiza la reversión: ").strip() or "usuario"
why=input("Motivo de la reversión: ").strip() or "restauración manual"
rec={"timestamp":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
     "version":choice,"who":who,"reason":why}
LOG.parent.mkdir(exist_ok=True); LOG.open("a",encoding="utf-8").write(json.dumps(rec)+"\n")
print("Rollback completado:",rec)
