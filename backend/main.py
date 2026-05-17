from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
import sys
from pathlib import Path

# Add backend directory to sys.path to resolve imports correctly
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from inference import BrainMRIPredictor, HEATMAP_DIR

app = FastAPI(
    title="Brain MRI Tumor API",
    description="Eğitim amaçlı beyin MR sınıflandırma + ısı haritası çıktısı. Tıbbi tanı yerine geçmez.",
    version="0.1.0",
)

# Production CORS settings
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://brain-mri-project-production.up.railway.app",
    "https://brain-mri-project.vercel.app",
    "https://brain-mri-project.onrender.com",
    "*"  # Geçici olarak tüm originlere izin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

predictor = None
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


@app.on_event("startup")
def _load_model():
    global predictor
    try:
        predictor = BrainMRIPredictor()
        print("[OK] Model yuklendi")
    except Exception as e:
        print(f"[ERROR] Model yuklenemedi: {e}")
        predictor = None
        print(f"Model load failed: {e}")


@app.post("/predict")
async def predict_mri(image: UploadFile = File(...), profile: str = Form(None)):
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Sadece JPEG/PNG destekleniyor.")
    if predictor is None:
        raise HTTPException(
            status_code=500,
            detail="Model yüklenemedi. Önce eğitimi çalıştırıp models/brain_mri_classifier.pth üret.",
        )
    image_bytes = await image.read()
    # Parse optional clinical profile JSON string
    clinical_profile = None
    if profile:
        try:
            import json
            clinical_profile = json.loads(profile)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Profil JSON ayrıştırma hatası: {e}")
    try:
        result = predictor.predict(image_bytes, clinical_profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    result["heatmap_url"] = f"/heatmaps/{result['heatmap_filename']}"
    return result


@app.get("/")
@app.get("/index.html")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/mrtaramalari.html")
def mr_taramalari(request: Request):
    return templates.TemplateResponse(request, "mrtaramalari.html", {})


@app.get("/analiz.html")
def analiz(request: Request):
    return templates.TemplateResponse(request, "analiz.html", {})


@app.get("/hastahikayesi.html")
def hasta_hikayesi(request: Request):
    return templates.TemplateResponse(request, "hastahikayesi.html", {})


@app.get("/hakkinda.html")
def hakkinda(request: Request):
    return templates.TemplateResponse(request, "hakkinda.html", {})


@app.get("/beyin.jpg")
def get_beyin_jpg():
    path = Path(__file__).resolve().parent / "templates" / "beyin.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path)


@app.get("/doktor.jpg")
def get_doktor_jpg():
    path = Path(__file__).resolve().parent / "templates" / "doktor.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path)


@app.get("/heatmaps/{filename}")
def get_heatmap(filename: str):
    path = HEATMAP_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Heatmap bulunamadı.")
    return FileResponse(path)


@app.get("/original/{filename}")
def get_original(filename: str):
    path = HEATMAP_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Orijinal görüntü bulunamadı.")
    return FileResponse(path)


