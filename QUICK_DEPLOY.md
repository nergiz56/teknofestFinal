# 🚀 HIZLI DEPLOYMENT

## ⚡ **5 DAKİKADA YAYINLAMA**

### 📦 **GitHub Reposu Oluştur**
1. GitHub hesabını aç
2. "New repository" → "brain-mri-project"
3. Public seç (ücretsiz için)
4. "Create repository"

### 🔗 **Local'i GitHub'a Bağla**
```bash
# Terminalde çalıştır (proje klasöründe):
git remote add origin https://github.com/[KULLANICIADI]/brain-mri-project.git
git push -u origin main
```

### 🌐 **Render.com Deploy**
1. [render.com](https://render.com) → GitHub ile giriş
2. **"New +" → "Web Service"**
3. GitHub reposunu seç
4. **Build Command**: `pip install -r requirements-prod.txt`
5. **Start Command**: `gunicorn backend.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`
6. **"Create Web Service"**

### ⏱️ **Bekle (2-3 dakika)**
- Build process otomatik başlar
- Model yüklenir
- Uygulama hazır olur

### 🎯 **Test Et**
- **Link**: `https://brain-mri-project.onrender.com`
- **MRI yükle**: Test et
- **Sonuçları kontrol et**

---

## 🚨 **HATA ÇÖZÜMÜ**

### Build Hatası:
- Model dosyası eksik → `models/brain_mri_classifier.pth` kontrol et
- Dependencies hatası → `requirements-prod.txt` kontrol et

### Runtime Hatası:
- Memory error → Render free plan yetersiz
- Model yüklenemez → Dosya yolu kontrol et

### CORS Hatası:
- Tarayıcı konsolunu kontrol et
- Origin ayarlarını kontrol et

---

## 📱 **JÜRİ İÇİN HAZIRLIK**

### Link Paylaşımı:
- **Ana link**: `https://brain-mri-project.onrender.com`
- **QR kod**: Linki QR'ya çevir
- **Yedek**: GitHub Pages alternatifi

### Test Kontrol Listesi:
- [ ] Sayfa açılıyor
- [ ] MRI yükleme çalışıyor
- [ ] Analiz sonuçları geliyor
- [ ] Tümör işaretleme doğru
- [ ] Tedavi önerileri detaylı

---

**🎉 Başarılı! Jüri linkini paylaşmaya hazır!**
