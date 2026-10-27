
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
    # Create DataFrame matching the trained feature name
    input_data = pd.DataFrame({
        "StudyHours": [ds.StudyHours]
    })

    # Predict marks
    marks = model.predict(input_data)

    return {
        "predictionmarks": round(float(marks[0]), 2)
    }
