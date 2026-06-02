import json

import numpy as np
import onnxruntime as ort
from fastapi import HTTPException, FastAPI, Request
from pydantic import BaseModel, Field, ConfigDict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

app = FastAPI(title="ONNX Regression API", version="1.0.0")
templates = Jinja2Templates(directory="web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_scaling_params(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        params = json.load(f)
    return np.array(params["mean"], dtype=np.float32), np.array(params["std"], dtype=np.float32)

# Load parameters for each model corresponding to their scaling files
mean2, std2 = load_scaling_params("ONNX student score predictor/scaling_params12.json")
mean3, std3 = load_scaling_params("ONNX student score predictor/scaling_params21.json")
mean4, std4 = load_scaling_params("ONNX student score predictor/scaling_params22.json")
mean5, std5 = load_scaling_params("ONNX student score predictor/scaling_params31.json")
mean6, std6 = load_scaling_params("ONNX student score predictor/scaling_params32.json")
mean7, std7 = load_scaling_params("ONNX student score predictor/scaling_params41.json")

MODEL2_PATH = "ONNX student score predictor/LGBMModel12.onnx"
MODEL3_PATH = "ONNX student score predictor/LGBMModel21.onnx"
MODEL4_PATH = "ONNX student score predictor/LGBMModel22.onnx"
MODEL5_PATH = "ONNX student score predictor/LGBMModel31.onnx"
MODEL6_PATH = "ONNX student score predictor/LGBMModel32.onnx"
MODEL7_PATH = "ONNX student score predictor/LGBMModel41.onnx"

session2 = ort.InferenceSession(MODEL2_PATH, providers=["CPUExecutionProvider"])
session3 = ort.InferenceSession(MODEL3_PATH, providers=["CPUExecutionProvider"])
session4 = ort.InferenceSession(MODEL4_PATH, providers=["CPUExecutionProvider"])
session5 = ort.InferenceSession(MODEL5_PATH, providers=["CPUExecutionProvider"])
session6 = ort.InferenceSession(MODEL6_PATH, providers=["CPUExecutionProvider"])
session7 = ort.InferenceSession(MODEL7_PATH, providers=["CPUExecutionProvider"])

input2_name = session2.get_inputs()[0].name
input3_name = session3.get_inputs()[0].name
input4_name = session4.get_inputs()[0].name
input5_name = session5.get_inputs()[0].name
input6_name = session6.get_inputs()[0].name
input7_name = session7.get_inputs()[0].name

output2_name = session2.get_outputs()[0].name
output3_name = session3.get_outputs()[0].name
output4_name = session4.get_outputs()[0].name
output5_name = session5.get_outputs()[0].name
output6_name = session6.get_outputs()[0].name
output7_name = session7.get_outputs()[0].name

class PredictRequest2(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    diploma_avg: float = Field(..., alias="diploma_avg")
    basic_science_avg: float = Field(..., alias="basic_science_avg")
    physiopathology_avg: float = Field(..., alias="physiopathology_avg")
    clinical_avg: float = Field(..., alias="clinical_avg")
    internship_avg: float = Field(..., alias="internship_avg")
    medical_course_type: float = Field(..., alias="medical_course_type")
    f11: float = Field(..., alias="11")

class PredictRequest3(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    diploma_avg: float = Field(..., alias="diploma_avg")
    basic_science_avg: float = Field(..., alias="basic_science_avg")
    physiopathology_avg: float = Field(..., alias="physiopathology_avg")
    clinical_avg: float = Field(..., alias="clinical_avg")
    internship_avg: float = Field(..., alias="internship_avg")
    medical_course_type: float = Field(..., alias="medical_course_type")
    f11: float = Field(..., alias="11")
    f12: float = Field(..., alias="12")

class PredictRequest4(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    diploma_avg: float = Field(..., alias="diploma_avg")
    basic_science_avg: float = Field(..., alias="basic_science_avg")
    physiopathology_avg: float = Field(..., alias="physiopathology_avg")
    clinical_avg: float = Field(..., alias="clinical_avg")
    internship_avg: float = Field(..., alias="internship_avg")
    medical_course_type: float = Field(..., alias="medical_course_type")
    f11: float = Field(..., alias="11")
    f12: float = Field(..., alias="12")
    f21: float = Field(..., alias="21")

class PredictRequest5(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    diploma_avg: float = Field(..., alias="diploma_avg")
    basic_science_avg: float = Field(..., alias="basic_science_avg")
    physiopathology_avg: float = Field(..., alias="physiopathology_avg")
    clinical_avg: float = Field(..., alias="clinical_avg")
    internship_avg: float = Field(..., alias="internship_avg")
    medical_course_type: float = Field(..., alias="medical_course_type")
    f11: float = Field(..., alias="11")
    f12: float = Field(..., alias="12")
    f21: float = Field(..., alias="21")
    f22: float = Field(..., alias="22")

class PredictRequest6(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    diploma_avg: float = Field(..., alias="diploma_avg")
    basic_science_avg: float = Field(..., alias="basic_science_avg")
    physiopathology_avg: float = Field(..., alias="physiopathology_avg")
    clinical_avg: float = Field(..., alias="clinical_avg")
    internship_avg: float = Field(..., alias="internship_avg")
    medical_course_type: float = Field(..., alias="medical_course_type")
    f11: float = Field(..., alias="11")
    f12: float = Field(..., alias="12")
    f21: float = Field(..., alias="21")
    f22: float = Field(..., alias="22")
    f31: float = Field(..., alias="31")

class PredictRequest7(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    diploma_avg: float = Field(..., alias="diploma_avg")
    basic_science_avg: float = Field(..., alias="basic_science_avg")
    physiopathology_avg: float = Field(..., alias="physiopathology_avg")
    clinical_avg: float = Field(..., alias="clinical_avg")
    internship_avg: float = Field(..., alias="internship_avg")
    medical_course_type: float = Field(..., alias="medical_course_type")
    f11: float = Field(..., alias="11")
    f12: float = Field(..., alias="12")
    f21: float = Field(..., alias="21")
    f22: float = Field(..., alias="22")
    f31: float = Field(..., alias="31")
    f32: float = Field(..., alias="32")


class PredictResponse(BaseModel):
    prediction: float

@app.get("/")
def serve_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict_2", response_model=PredictResponse)
def predict2(data: PredictRequest2):
    try:
        features = np.array([[
            data.diploma_avg,
            data.basic_science_avg,
            data.physiopathology_avg,
            data.clinical_avg,
            data.internship_avg,
            data.medical_course_type,
            data.f11,
        ]], dtype=np.float32)
        scaled_features = (features - mean2) / std2
        result = session2.run([output2_name], {input2_name: scaled_features})[0]
        prediction = float(result[0][0] if result.ndim == 2 else result[0])

        return PredictResponse(prediction=prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/predict_3", response_model=PredictResponse)
def predict3(data: PredictRequest3):
    try:
        features = np.array([[
            data.diploma_avg,
            data.basic_science_avg,
            data.physiopathology_avg,
            data.clinical_avg,
            data.internship_avg,
            data.medical_course_type,
            data.f11,
            data.f12,
        ]], dtype=np.float32)
        scaled_features = (features - mean3) / std3
        result = session3.run([output3_name], {input3_name: scaled_features})[0]
        prediction = float(result[0][0] if result.ndim == 2 else result[0])

        return PredictResponse(prediction=prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/predict_4", response_model=PredictResponse)
def predict4(data: PredictRequest4):
    try:
        # Keep the exact feature order used during training
        features = np.array([[
            data.diploma_avg,
            data.basic_science_avg,
            data.physiopathology_avg,
            data.clinical_avg,
            data.internship_avg,
            data.medical_course_type,
            data.f11,
            data.f12,
            data.f21,
        ]], dtype=np.float32)
        scaled_features = (features - mean4) / std4
        result = session4.run([output4_name], {input4_name: scaled_features})[0]
        prediction = float(result[0][0] if result.ndim == 2 else result[0])

        return PredictResponse(prediction=prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/predict_5", response_model=PredictResponse)
def predict5(data: PredictRequest5):
    try:
        features = np.array([[
            data.diploma_avg,
            data.basic_science_avg,
            data.physiopathology_avg,
            data.clinical_avg,
            data.internship_avg,
            data.medical_course_type,
            data.f11,
            data.f12,
            data.f21,
            data.f22,
        ]], dtype=np.float32)
        scaled_features = (features - mean5) / std5
        result = session5.run([output5_name], {input5_name: scaled_features})[0]
        prediction = float(result[0][0] if result.ndim == 2 else result[0])

        return PredictResponse(prediction=prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/predict_6", response_model=PredictResponse)
def predict6(data: PredictRequest6):
    try:
        features = np.array([[
            data.diploma_avg,
            data.basic_science_avg,
            data.physiopathology_avg,
            data.clinical_avg,
            data.internship_avg,
            data.medical_course_type,
            data.f11,
            data.f12,
            data.f21,
            data.f22,
            data.f31,
        ]], dtype=np.float32)
        scaled_features = (features - mean6) / std6
        result = session6.run([output6_name], {input6_name: scaled_features})[0]
        prediction = float(result[0][0] if result.ndim == 2 else result[0])

        return PredictResponse(prediction=prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/predict_7", response_model=PredictResponse)
def predict7(data: PredictRequest7):
    try:
        features = np.array([[
            data.diploma_avg,
            data.basic_science_avg,
            data.physiopathology_avg,
            data.clinical_avg,
            data.internship_avg,
            data.medical_course_type,
            data.f11,
            data.f12,
            data.f21,
            data.f22,
            data.f31,
            data.f32,
        ]], dtype=np.float32)
        scaled_features = (features - mean7) / std7
        result = session7.run([output7_name], {input7_name: scaled_features})[0]
        prediction = float(result[0][0] if result.ndim == 2 else result[0])

        return PredictResponse(prediction=prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
