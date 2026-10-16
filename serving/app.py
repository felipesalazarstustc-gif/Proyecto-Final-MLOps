import json, os, time, threading
from pathlib import Path
import mlflow, pandas as pd
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient

app=Flask(__name__)
ROOT=Path(__file__).resolve().parents[1]
MODEL_NAME=os.getenv("MLFLOW_MODEL_NAME","OrvexaRegressor")
TRACK=os.getenv("MLFLOW_TRACKING_URI","http://mlflow:5000")
LOCK=threading.Lock()
STATS={"predictions":0,"errors":0,"latencies_ms":[]}

EXPECTED=["timestamp","Hour","TemperatureC","Humidity","Wind_speed_ms","Visibility10m",
"Dew_point_temperatureC","Solar_RadiationMJm2","Rainfallmm","Snowfall_cm","Seasons",
"Holiday","Functioning_Day","year","month_num","day","weekday_num"]

def get_model():
    mlflow.set_tracking_uri(TRACK)
    c=MlflowClient()
    versions=c.get_latest_versions(MODEL_NAME,stages=["Production"])
    if not versions: raise RuntimeError("No hay modelo en Production")
    v=versions[0]
    model=mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
    return model,str(v.version)

def log_prediction(inp,pred,version):
    p=ROOT/"evidencias/predictions.jsonl"; p.parent.mkdir(exist_ok=True)
    rec={"timestamp":time.time(),"input":inp,"prediction":float(pred),"model_version":version}
    with p.open("a",encoding="utf-8") as f: f.write(json.dumps(rec)+"\n")

def validate_one(x):
    if not isinstance(x,dict): return "body must be an object"
    missing=[c for c in EXPECTED if c not in x]
    if missing: return f"missing columns: {missing}"
    return None

@app.get("/health")
def health():
    try:
        _,v=get_model()
        return jsonify({"status":"ok","model_name":MODEL_NAME,"model_version":v})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}),503

@app.get("/metrics")
def metrics():
    with LOCK:
        l=STATS["latencies_ms"]
        return jsonify({"predictions_total":STATS["predictions"],
                        "errors_total":STATS["errors"],
                        "latency_ms_avg":sum(l)/len(l) if l else 0})

@app.post("/predict")
def predict():
    x=request.get_json(silent=True)
    err=validate_one(x)
    if err:
        STATS["errors"]+=1; return jsonify({"error":err}),400
    try:
        t=time.perf_counter(); model,v=get_model()
        pred=model.predict(pd.DataFrame([x]))[0]
        ms=(time.perf_counter()-t)*1000
        with LOCK: STATS["predictions"]+=1; STATS["latencies_ms"].append(ms)
        log_prediction(x,pred,v)
        return jsonify({"prediction":float(pred),"model_version":v,"latency_ms":ms})
    except Exception as e:
        STATS["errors"]+=1; return jsonify({"error":str(e)}),500

@app.post("/predict/batch")
def predict_batch():
    data=request.get_json(silent=True)
    if not isinstance(data,list) or not data: return jsonify({"error":"body must be a non-empty list"}),400
    for x in data:
        err=validate_one(x)
        if err: STATS["errors"]+=1; return jsonify({"error":err}),400
    try:
        t=time.perf_counter(); model,v=get_model()
        preds=model.predict(pd.DataFrame(data))
        ms=(time.perf_counter()-t)*1000
        with LOCK: STATS["predictions"]+=len(data); STATS["latencies_ms"].append(ms/len(data))
        for x,p in zip(data,preds): log_prediction(x,p,v)
        return jsonify({"predictions":[float(p) for p in preds],"model_version":v,"latency_ms_total":ms})
    except Exception as e:
        STATS["errors"]+=1; return jsonify({"error":str(e)}),500

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)
