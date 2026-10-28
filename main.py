from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("linearmodel.pkl")


class DataScience(BaseModel):
    StudyHours: float


@app.get("/")
def test():
    return {"msg": "uvicorn is running"}


@app.post("/predict")
def datainsert(ds: DataScience):
    marks = model.predict([[ds.StudyHours]])
    return {"predicted_marks": round(float(marks[0]), 2)}