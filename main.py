from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os

from model import wheat_service

app = FastAPI(
    title="Wheat Seed Variety Classification API",
    description="PyTorch Derin Öğrenme Modeli ile Buğday Tohumu Varyete Tespiti ve Sınıflandırma Servisi",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Mount Static & Templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Pydantic Input Schema
class WheatFeaturesInput(BaseModel):
    area: float = Field(..., ge=8.0, le=25.0, description="Tohum Alanı (A - mm²)", example=15.26)
    perimeter: float = Field(..., ge=10.0, le=20.0, description="Tohum Çevresi (P - mm)", example=14.84)
    compactness: float = Field(..., ge=0.70, le=0.98, description="Kompaklık (C = 4*pi*A/P²)", example=0.871)
    length_of_kernel: float = Field(..., ge=4.0, le=7.5, description="Tane Uzunluğu (mm)", example=5.763)
    width_of_kernel: float = Field(..., ge=2.0, le=4.5, description="Tane Genişliği (mm)", example=3.312)
    asymmetry_coefficient: float = Field(..., ge=0.5, le=9.5, description="Asimetri Katsayısı", example=2.221)
    length_of_groove: float = Field(..., ge=4.0, le=7.0, description="Tane Oluk Uzunluğu (mm)", example=5.22)

# Presets for quick UI testing
PRESETS = {
    "kama": {
        "name": "Kama Buğdayı Örneği",
        "values": {
            "area": 14.88,
            "perimeter": 14.57,
            "compactness": 0.8811,
            "length_of_kernel": 5.554,
            "width_of_kernel": 3.333,
            "asymmetry_coefficient": 1.018,
            "length_of_groove": 4.956
        }
    },
    "rosa": {
        "name": "Rosa Buğdayı Örneği",
        "values": {
            "area": 19.13,
            "perimeter": 16.53,
            "compactness": 0.8810,
            "length_of_kernel": 6.448,
            "width_of_kernel": 3.696,
            "asymmetry_coefficient": 3.242,
            "length_of_groove": 6.421
        }
    },
    "canadian": {
        "name": "Canadian Buğdayı Örneği",
        "values": {
            "area": 11.84,
            "perimeter": 13.21,
            "compactness": 0.8529,
            "length_of_kernel": 5.175,
            "width_of_kernel": 2.836,
            "asymmetry_coefficient": 3.598,
            "length_of_groove": 5.044
        }
    }
}

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/predict")
async def predict_variety(data: WheatFeaturesInput):
    try:
        features = [
            data.area,
            data.perimeter,
            data.compactness,
            data.length_of_kernel,
            data.width_of_kernel,
            data.asymmetry_coefficient,
            data.length_of_groove
        ]
        result = wheat_service.predict(features)
        return JSONResponse(status_code=200, content={"success": True, "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/api/presets")
async def get_presets():
    return JSONResponse(status_code=200, content=PRESETS)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "WheatSeedsClassifier",
        "framework": "PyTorch + FastAPI",
        "classes": list(wheat_service.labels.values())
    }
