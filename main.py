from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model
model = joblib.load("linearmodel.pkl")

class datascience(BaseModel):
    StudyHours: float

@app.get("/")
def test():
    return {"msg": "uvicorn is running"}

@app.post("/predict")
def datainsert(ds: datascience):
    # Input DataFrame matching the trained feature name
    marks = model.predict({"StudyHours": [ds.StudyHours]})
    
    return {"predictionmarks": round(float(marks[0]), 2)}
