import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
import joblib, mlflow, mlflow.sklearn, numpy as np, pandas as pd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT/"data/batches"
ART = ROOT/"data/artifacts"
ART.mkdir(parents=True, exist_ok=True)
TARGET="target"
DROP=["timestamp","target"]
CATEGORICAL=["Seasons","Holiday","Functioning_Day"]
NUMERIC=["Hour","TemperatureC","Humidity","Wind_speed_ms","Visibility10m",
         "Dew_point_temperatureC","Solar_RadiationMJm2","Rainfallmm","Snowfall_cm",
         "year","month_num","day","weekday_num"]

def file_hash(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def git_hash():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception: return "NO_GIT_COMMIT"

def metrics(y,p):
    rmse=float(np.sqrt(mean_squared_error(y,p)))
    mae=float(mean_absolute_error(y,p))
    mape=float(np.mean(np.abs((np.asarray(y)-np.asarray(p))/np.maximum(np.abs(y),1e-8)))*100)
    return rmse,mae,mape

def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                          ("oh",OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL)
    ])

def build_model(family, p):
    if family=="ridge":
        return Ridge(alpha=float(p["alpha"]))
    if family=="gbr":
        return GradientBoostingRegressor(
            n_estimators=int(p["n_estimators"]),
            learning_rate=float(p["learning_rate"]),
            max_depth=int(p["max_depth"]),
            random_state=42
        )
    raise ValueError(family)

def run_campaign(batch="L0", max_evals=30):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000"))
    mlflow.set_experiment("Variant_A_Regression")
    df=pd.read_csv(BATCH_DIR/f"{batch}.csv")
    # Temporal validation: last 20% of the batch.
    split=int(len(df)*0.8)
    tr,va=df.iloc[:split],df.iloc[split:]
    Xtr,ytr=tr.drop(columns=[TARGET]),tr[TARGET]
    Xva,yva=va.drop(columns=[TARGET]),va[TARGET]
    dataset_hash=file_hash(BATCH_DIR/f"{batch}.csv")
    families=["ridge","gbr"]
    per_family=max(15,max_evals//2)
    all_results=[]
    for family in families:
        if family=="ridge":
            space={"alpha":hp.loguniform("alpha",np.log(0.01),np.log(100))}
        else:
            space={
                "n_estimators":hp.quniform("n_estimators",50,250,10),
                "learning_rate":hp.loguniform("learning_rate",np.log(0.01),np.log(0.2)),
                "max_depth":hp.quniform("max_depth",2,8,1)
            }
        trials=Trials()
        def objective(p):
            start=time.perf_counter()
            model=Pipeline([("prep",make_preprocessor()),("model",build_model(family,p))])
            model.fit(Xtr,ytr)
            pred=model.predict(Xva)
            elapsed=(time.perf_counter()-start)*1000/len(Xva)
            rmse,mae,mape=metrics(yva,pred)
            return {"loss":rmse,"status":STATUS_OK,"mae":mae,"mape":mape,"ms_obs":elapsed}
        best=fmin(objective,space,algo=tpe.suggest,max_evals=per_family,trials=trials,rstate=np.random.default_rng(42))
        for i,res in enumerate(trials.results):
            all_results.append((family,i,res))
            params=trials.trials[i]["misc"]["vals"]
            if family=="ridge":
                hpv={"alpha":float(params["alpha"][0])}
            else:
                hpv={
                    "n_estimators":int(params["n_estimators"][0]),
                    "learning_rate":float(params["learning_rate"][0]),
                    "max_depth":int(params["max_depth"][0])
                }
            with mlflow.start_run(run_name=f"{family}_{batch}_{i:02d}") as run:
                model=Pipeline([("prep",make_preprocessor()),("model",build_model(family,hpv))])
                t0=time.perf_counter(); model.fit(Xtr,ytr); train_s=time.perf_counter()-t0
                pred=model.predict(Xva)
                rmse,mae,mape=metrics(yva,pred)
                mlflow.log_params(hpv)
                mlflow.log_metrics({"rmse_val":rmse,"mae_val":mae,"mape_val":mape,
                                    "train_seconds":train_s,"inference_ms_per_observation":res["ms_obs"]})
                mlflow.set_tags({
                    "model_family":family,"batch":batch,"dataset_hash":dataset_hash,
                    "code_commit":git_hash(),"target":TARGET,"variant":"A"
                })
                model_path=ART/f"{run.info.run_id}.joblib"
                joblib.dump(model,model_path)
                mlflow.log_artifact(str(model_path),artifact_path="model")
                vars_path=ART/f"{run.info.run_id}_variables.json"
                vars_path.write_text(json.dumps({"numeric":NUMERIC,"categorical":CATEGORICAL},indent=2))
                mlflow.log_artifact(str(vars_path),artifact_path="metadata")
                all_results[-1]=all_results[-1]+(run.info.run_id,)
    best=sorted(all_results,key=lambda x:x[2]["loss"])[0]
    summary={"best_family":best[0],"best_rmse":best[2]["loss"],"runs":len(all_results)}
    (ART/f"campaign_{batch}.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

def train_single(batch, family="gbr", params=None, run_name="retrain"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000"))
    mlflow.set_experiment("Variant_A_Retraining")
    df=pd.read_csv(BATCH_DIR/f"{batch}.csv")
    split=int(len(df)*0.8)
    tr,va=df.iloc[:split],df.iloc[split:]
    Xtr,ytr=tr.drop(columns=[TARGET]),tr[TARGET]
    Xva,yva=va.drop(columns=[TARGET]),va[TARGET]
    if params is None:
        params={"n_estimators":120,"learning_rate":0.05,"max_depth":4}
    with mlflow.start_run(run_name=run_name) as run:
        model=Pipeline([("prep",make_preprocessor()),("model",build_model(family,params))])
        t0=time.perf_counter(); model.fit(Xtr,ytr); train_s=time.perf_counter()-t0
        t1=time.perf_counter(); pred=model.predict(Xva); ms=(time.perf_counter()-t1)*1000/len(Xva)
        rmse,mae,mape=metrics(yva,pred)
        ds=file_hash(BATCH_DIR/f"{batch}.csv")
        mlflow.log_params(params)
        mlflow.log_metrics({"rmse_val":rmse,"mae_val":mae,"mape_val":mape,
                            "train_seconds":train_s,"inference_ms_per_observation":ms})
        mlflow.set_tags({"model_family":family,"batch":batch,"dataset_hash":ds,
                         "code_commit":git_hash(),"variant":"A"})
        path=ART/f"{run.info.run_id}.joblib"; joblib.dump(model,path)
        mlflow.log_artifact(str(path),artifact_path="model")
        model_uri=f"runs:/{run.info.run_id}/model/{path.name}"
        # Register directly; promotion script controls Production.
        name=os.getenv("MLFLOW_MODEL_NAME","OrvexaRegressor")
        try: mlflow.register_model(model_uri,name)
        except Exception as e: print("Registry warning:",e)
        return {"run_id":run.info.run_id,"rmse":rmse,"mae":mae,"mape":mape,"ms_obs":ms,
                "model_uri":model_uri}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--campaign",action="store_true")
    ap.add_argument("--batch",default="L0")
    ap.add_argument("--max-evals",type=int,default=30)
    args=ap.parse_args()
    if args.campaign: run_campaign(args.batch,args.max_evals)
    else: print(train_single(args.batch))
