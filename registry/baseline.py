import os, sys, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.train import train_single
import mlflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000"))
# Train a fixed baseline and mark it. This is intentionally simple and reproducible.
import mlflow
with mlflow.start_run(run_name="baseline_L0") as run:
    # train_single creates its own run, so we only use its result below.
    pass
res=train_single("L0",family="ridge",params={"alpha":1.0},run_name="baseline_L0_actual")
from mlflow.tracking import MlflowClient
c=MlflowClient()
c.set_tag(res["run_id"],"baseline","true")
(ROOT/"data/artifacts").mkdir(parents=True,exist_ok=True)
(ROOT/"data/artifacts/baseline.json").write_text(json.dumps({
    "run_id":res["run_id"],"rmse":res["rmse"],"ms_obs":res["ms_obs"]},indent=2))
print(res)
