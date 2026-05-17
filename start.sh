#!/bin/bash

# Brain MRI Project Deployment Script

echo "🚀 Brain MRI Project Deployment Başlatılıyor..."

# 1. GitHub setup
echo "📦 GitHub reposu hazırlanıyor..."
git init
git add .
git commit -m "Initial commit - Brain MRI Tumor Detection"
git branch -M main

# 2. Environment setup
echo "🔧 Python environment hazırlanıyor..."
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-prod.txt

# 3. Model check
echo "🧠 Model kontrol ediliyor..."
if [ ! -f "models/brain_mri_classifier.pth" ]; then
    echo "⚠️ Model dosyası bulunamadı! Lütfen eğitim scriptini çalıştırın."
    echo "📝 Çalıştır: python training/train_classifier.py"
    exit 1
fi

echo "✅ Deployment hazır!"
echo ""
echo "📋 Sonraki adımlar:"
echo "1. GitHub'a push yap: git remote add origin <repo-url>"
echo "2. Render.com'a git ve reposunu import et"
echo "3. Build settings'i yap"
echo "4. Deploy et!"
echo ""
echo "🌐 Uygulama linki: https://brain-mri-project.onrender.com"
