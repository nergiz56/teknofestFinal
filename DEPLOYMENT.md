# 🚀 Brain MRI Project Deployment Guide

## 📋 Hızlı Deployment (Render.com)

### 1. GitHub'a Yükle
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/kullaniciadi/brain-mri-project.git
git push -u origin main
```

### 2. Render.com'a Git
1. [render.com](https://render.com) sitesine git
2. GitHub ile hesap aç
3. "New +" → "Web Service"
4. GitHub reposunu seç
5. Ayarları yap:

**Build Settings:**
- Build Command: `pip install -r requirements-prod.txt`
- Start Command: `gunicorn backend.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`

**Environment:**
- Python: 3.11
- Region: Frankfurt (Avrupa için hızlı)
- Instance Type: Free

### 3. Domain Ayarları
- Render domain: `brain-mri-project.onrender.com`
- Custom domain: Kendi domainini ekle

## 🔧 Alternatif Platformlar

### Vercel (Frontend odaklı)
```bash
# vercel.json oluştur
{
  "functions": {
    "api/*.py": {
      "runtime": "python3.9"
    }
  }
}
```

### PythonAnywhere
```bash
# Web tabında dosyaları yükle
# Virtual environment oluştur
pip install -r requirements-prod.txt
# WSGI dosyasını yapılandır
```

## 📱 Jüri Erişimi

### Link Paylaşımı
1. **Render link**: `https://brain-mri-project.onrender.com`
2. **QR kod**: Linki QR koda çevir
3. **Demo videosu**: Nasıl kullanılacağını göster

### Erişim Kontrolü
- **Public**: Herkes erişebilir
- **Password**: Basic auth ile koruma
- **IP limit**: Belirli IP'lere erişim

## 🛠️ Production Optimizasyonları

### Model Optimizasyonu
```python
# Model boyutunu küçült
torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

### Cache Ayarları
```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend

FastAPICache.init(InMemoryBackend())
```

### Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("5/minute")
async def predict_mri(request: Request, image: UploadFile = File(...)):
    # ...
```

## 📊 Monitoring

### Loglama
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict_mri(image: UploadFile = File(...)):
    logger.info(f"Prediction request for image: {image.filename}")
    # ...
```

### Health Check
```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": predictor is not None}
```

## 🚨 Hata Çözümü

### Yaygın Sorunlar
1. **Model yüklenemez**: Model dosyasını doğru yere koy
2. **Memory error**: Free plan yetersiz → upgrade et
3. **CORS error**: Origin listesini güncelle
4. **Slow response**: Model optimizasyonu yap

### Debug Tips
```python
# Debug mode
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

## 🎯 Başarı Kontrolü

### Test Edilecekler
- [ ] Ana sayfa açılıyor mu?
- [ ] MRI yükleme çalışıyor mu?
- [ ] Sonuçlar doğru gösteriliyor mu?
- [ ] Heatmap oluşuyor mu?
- [ ] Mobil uyumlu mu?

### Jüri İçin Hazırlık
1. **Demo verisi**: Test MR görüntüleri
2. **Kullanım kılavuzu**: Basit açıklamalar
3. **İletişim**: Sorular için mail/telefon
4. **Yedek link**: Alternatif erişim

## 📞 Destek

Sorun olursa:
- GitHub Issues
- Render docs
- FastAPI documentation
- Discord community

---

**Not**: Bu demo eğitim amaçlıdır. Tıbbi kullanım için uygun değildir.
