#!/bin/bash
# Railway.app Hızlı Deploy Script

echo "🚀 Railway.app Deploy Başlatılıyor..."

# 1. Railway CLI kur
npm install -g @railway/cli

# 2. Giriş yap
railway login

# 3. Proje oluştur
railway new

# 4. GitHub reposunu bağla
railway variables set PYTHON_VERSION 3.11
railway variables set PORT 8000

# 5. Deploy et
railway up

echo "✅ Railway deploy tamamlandı!"
echo "🌐 URL: https://brain-mri-project-production.up.railway.app"
