# 🧠 Brain MRI Tumor Classification & Clinical Decision Support System

> **Çok Katmanlı Derin Öğrenme (PyTorch ResNet18), Görsel Segmentasyon (Grad-CAM) ve Klinik Karar Destek Hattını (Dual-XGBoost) Birleştiren Hibrit Teşhis Platformu**

---

## 📋 Projeye Genel Bakış

Bu platform; beyin MR (Manyetik Rezonans) görüntülerinden tümör varlığını ve türünü yüksek doğrulukla saptayan **Evrişimsel Sinir Ağı (CNN - ResNet18)** tabanlı derin öğrenme modeli ile, hastanın klinik ve genetik profilini inceleyen **XGBoost makine öğrenmesi modellerini** hibrit bir şekilde entegre eder. 

Sistem, Grad-CAM (Gradient-weighted Class Activation Mapping) kullanarak tümörün beyindeki konumunu piksel bazında tespit edip görselleştirir ve hekimlere yardımcı olabilecek **AI tabanlı tedavi ve takip tavsiyeleri** oluşturur.

---

## 📐 1. Teorik ve Teknik Detaylar (Yapay Zeka Mimarisi)

Sistem, iki ana yapay zeka hattı ve bir karar entegrasyon katmanından oluşur:

### A. Görüntü İşleme ve Derin Öğrenme Hattı (PyTorch & Grad-CAM)
1.  **Model Mimarisi (ResNet18):**
    *   MRI görüntülerindeki mekansal özellikleri ve doku örüntülerini yakalamak amacıyla **ResNet18** mimarisi kullanılmıştır.
    *   ResNet (Residual Network), derin ağlarda görülen **kaybolan gradyanlar (vanishing gradients)** problemini aşmak için **artık bağlantılar (skip connections / residual blocks)** kullanır. Bu sayede model, katmanlar derinleştikçe özellikleri bozmadan doğrudan sonraki katmanlara aktarabilir.
    *   Girdi boyutu olarak `224x224x3` (RGB) kabul edilir. Modelin son tam bağlantılı katmanı (fully connected layer), veri setindeki sınıflara (`glioma`, `menin`, `tumor`) göre 3 çıkış üretecek şekilde yeniden yapılandırılmıştır.

2.  **Grad-CAM (Gradient-weighted Class Activation Mapping) Mekanizması:**
    *   Modelin kararlarını açıklanabilir kılmak (**XAI - Explainable AI**) amacıyla son evrişimsel katman olan `layer4` üzerine **forward ve backward hook**'lar (kancalar) yerleştirilmiştir.
    *   Grad-CAM, hedef sınıfın ($c$) skorunun, seçilen evrişimsel katmanın aktivasyon haritasına ($A^k$) göre gradyanlarını hesaplar:
        $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}$$
    *   Burada hesaplanan $\alpha_k^c$ ağırlıkları, aktivasyon haritaları ile çarpılarak toplanır ve negatif etkileri yok etmek için ReLU fonksiyonundan geçirilir:
        $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$
    *   Elde edilen kaba aktivasyon haritası, OpenCV kullanılarak orijinal görüntü boyutuna çift doğrusal enterpolasyon (bilinear interpolation) yöntemiyle büyütülür.
    *   Aktivasyon skoru `0.6` ve üzerinde olan piksel kümeleri tespit edilerek tümör sınırları (**Bounding Box**) ve merkez koordinatları belirlenir. Orijinal görüntü üzerine jet-colormap formatında maskelenerek bindirilir (`cv2.addWeighted`).

---

### B. Klinik ve Genetik Profil Analiz Hattı (Dual-XGBoost Cascade)
Sadece görüntüye güvenmek yerine, hastanın moleküler profili ve genetik test sonuçları da analiz edilerek teşhis doğrulanır. Bunun için **iki aşamalı ardışık (cascade) XGBoost** modeli geliştirilmiştir:

1.  **Birinci Aşama (XGBoost Binary Classifier):**
    *   Hastanın genetik mutasyon durumları (IDH1, TP53, ATRX, PTEN, EGFR, CIC, MUC16 vb.), yaşı, cinsiyeti ve mutasyon yükü girdilerini alarak **Tümör Şüphesi Var/Yok** kararı üretir.
    *   Sınıf dengesizliği (class imbalance) problemi, eğitim sırasında `scale_pos_weight` parametresi optimize edilerek çözülmüştür.
2.  **İkinci Aşama (XGBoost Subtype Classifier):**
    *   İlk aşamada "Tümör Şüphesi" saptanan hastalar için devreye girer. Tümörün moleküler alt tipini (**GBM, LGG, SARCOMA, TUMOR**) tahmin eder.
    *   Model, çok sınıflı sınıflandırma (`multi:softmax`) modunda eğitilmiştir.

```mermaid
graph TD
    subgraph Görüntü İşleme (Derin Öğrenme)
        I1[MRI Görüntüsü] --> I2[Ön İşleme & Resize 224x224]
        I2 --> I3[ResNet18 CNN Modeli]
        I3 --> I4[Sınıf Skoru Tahmini]
        I3 --> I5[Grad-CAM layer4 Hook]
        I5 --> I6[Bilinear Interpolation]
        I6 --> I7[Tümör Isı Haritası & Konum Tespiti]
    end

    subgraph Klinik Analiz (Makine Öğrenmesi)
        C1[Yaş, Cinsiyet, Genler] --> C2[XGBoost İkili Sınıflandırıcı]
        C2 -->|Tümör Yok| C3[NORMAL Durum Raporu]
        C2 -->|Tümör Şüphesi| C4[XGBoost Çoklu Sınıflandırıcı]
        C4 --> C5[Tümör Alt Tip Teşhisi]
    end

    I7 --> Karar[Entegrasyon Katmanı & AI Tedavi Önerileri]
    C5 --> Karar
    C3 --> Karar
    I4 --> Karar
    Karar --> Rapor[Web Arayüzü Detaylı Analiz Raporu]
```

---

## 📁 2. Proje Klasör Yapısı ve Dosya Görevleri

```directory
behzat/
├── backend/
│   ├── main.py                 # FastAPI uygulamasının giriş noktası. CORS, Jinja2 şablonları ve API endpointleri burada tanımlıdır.
│   ├── inference.py            # PyTorch ve XGBoost tahmin motorunu barındıran çekirdek dosya. Grad-CAM algoritmalarını ve fail-safe sistemini yönetir.
│   └── templates/              # Kullanıcı arayüzünü (UI) oluşturan HTML ve CSS şablonları.
│       ├── index.html          # Ana görsel MR yükleme ve analiz paneli.
│       ├── analiz.html         # Grad-CAM ısı haritalarını, tümör boyutunu, evresini ve klinik profil analizini sunan gelişmiş rapor sayfası.
│       └── hakkinda.html       # Yapay zeka modelleri ve proje hakkında teknik bilgi veren doküman sayfası.
├── models/
│   ├── brain_mri_classifier.pth # Eğitilmiş PyTorch ResNet18 sinir ağı ağırlıkları.
│   ├── xgb_binary.pkl          # Klinik veri analizi yapan XGBoost ikili sınıflandırıcı modeli.
│   ├── xgb_subtype.pkl         # Tümör alt tipi belirleyen XGBoost çoklu sınıflandırıcı modeli.
│   └── heatmaps/               # Grad-CAM analizi sonrasında üretilen orijinal ve ısı haritalı çıktı görsellerinin kaydedildiği klasör.
├── training/                   # Model eğitim kodları ve veri seti hazırlama betikleri.
├── generate_mock_model.py      # Test amaçlı geçici (mock) PyTorch modeli oluşturan bağımsız yardımcı script.
├── requirements.txt            # Projenin çalışabilmesi için yüklenmesi gereken Python kütüphane listesi.
└── README.md                   # Şu an okumakta olduğunuz, tüm teknik detayları içeren ana proje dokümantasyonu.
```

---

## 🛠 3. Adım Adım Kurulum ve Çalıştırma Kılavuzu

Proje, hem local bilgisayarda hem de farklı sistemlerde sorunsuz çalışabilmesi için **çift çalışma moduna** sahiptir.

### A. Local Bilgisayarda Manuel Kurulum (Windows / macOS / Linux)

#### 1. Proje Klasörüne Giriş Yapın
Terminalinizi (PowerShell, Komut İstemi veya Terminal) açarak projenin ana klasörüne (`requirements.txt` dosyasının bulunduğu dizine) geçiş yapın:
```powershell
cd behzat
```

#### 2. Temiz Bir Sanal Ortam (venv) Oluşturun
İşletim sisteminizde kurulu olan Python sürümünü izole etmek amacıyla temiz bir sanal ortam oluşturun:
```powershell
python -m venv venv
```

#### 3. Bağımlılıkları Sanal Ortama Yükleyin
PowerShell script çalıştırma engellerine takılmamak adına, doğrudan sanal ortamın içindeki `pip` çalıştırıcısını çağırarak tüm gerekli kütüphaneleri kurun:
```powershell
venv\Scripts\pip install -r requirements.txt
```
*(macOS / Linux için: `source venv/bin/activate` yapıp ardından `pip install -r requirements.txt` çalıştırabilirsiniz).*

#### 4. Sunucuyu Başlatın
Uygulamayı çalıştırmak için `PYTHONPATH` tanımlaması yaparak FastAPI sunucusunu uvicorn ile başlatın:

*   **Windows (PowerShell) için:**
    ```powershell
    $env:PYTHONPATH="backend"; venv\Scripts\python -m uvicorn backend.main:app --reload --port 8000
    ```
*   **Windows (CMD / Komut İstemi) için:**
    ```cmd
    set PYTHONPATH=backend && venv\Scripts\python -m uvicorn backend.main:app --reload --port 8000
    ```
*   **macOS / Linux için:**
    ```bash
    PYTHONPATH=backend venv/bin/python -m uvicorn backend.main:app --reload --port 8000
    ```

#### 5. Arayüzü Açın
Sunucu başarıyla başladıktan sonra web tarayıcınızı açarak şu adrese gidin:
👉 **[http://localhost:8000](http://localhost:8000)**

---

### B. Docker ile Konteyner Tabanlı Kurulum (Alternatif Hızlı Kurulum)

Bilgisayarınızda Python kurulu olmasa dahi Docker yüklü ise projeyi tek bir komutla ayağa kaldırabilirsiniz.

1.  **Docker İmajını Derleyin:**
    ```bash
    docker build -t brain-mri-app .
    ```
2.  **Konteyneri Başlatın:**
    ```bash
    docker run -p 8000:8000 brain-mri-app
    ```
3.  Uygulamaya yine tarayıcınızdan **`http://localhost:8000`** adresinden erişebilirsiniz.

---

## 📦 4. ZIP Paketleme Kılavuzu (Başka PC'ye Gönderirken)

Projeyi başka bir bilgisayara (örneğin jüriye veya öğretmene) göndermek üzere sıkıştırırken şu kurallara kesinlikle uyun:

*   **KESİNLİKLE EKLENMEYECEK:** **`venv/`** ve **`__pycache__/`** klasörlerini ZIP dosyasına **dahil etmeyin.** `venv` klasörü sizin bilgisayarınızın işletim sistemine ve dosya yollarına göre kurulmuştur, başka bilgisayarda çalışmaz. Ayrıca ZIP dosyasının boyutunu gereksiz yere 3.5 GB büyüterek aktarımı imkansız hale getirir.
*   **MUTLAKA EKLENECEK:** `backend/`, `models/`, `requirements.txt` ve `README.md` dosyalarını seçerek sıkıştırın. ZIP boyutu sadece **~45 MB** olacaktır.

---

## 🌐 5. API Uç Noktaları (API Endpoints Reference)

Platform, harici entegrasyonlara izin veren güçlü bir RESTful API sunar:

### 1. Web Arayüzü (Ana Sayfa)
*   **URL:** `GET /` veya `GET /index.html`
*   **Görevi:** Kullanıcı arayüzünü (kontrol panelini) yükler.

### 2. MRI Analiz & Tahmin Servisi
*   **URL:** `POST /predict`
*   **İçerik Tipi (Content-Type):** `multipart/form-data`
*   **İstek Parametreleri:**
    *   `image` (File, Zorunlu): Analiz edilecek beyin MR görüntüsü (JPEG/PNG).
    *   `profile` (String, İsteğe Bağlı): Hasta klinik ve genetik profil verilerini içeren JSON string.
        *   Örnek Profil JSON formatı:
            ```json
            {"yas_kod": 45, "cinsiyet_kod": 1, "IDH1": 1, "TP53": 0, "ATRX": 1, "PTEN": 0}
            ```
*   **Yanıt Formatı (Response JSON Schema):**
    ```json
    {
      "tumor_present": true,
      "tumor_type": "glioma",
      "probabilities": {
        "glioma": 0.892,
        "menin": 0.081,
        "tumor": 0.027
      },
      "tumor_size_category": "medium",
      "stage_estimate": "intermediate",
      "treatment_suggestion": "🧬 Glioma (Orta Evre)\n\n📋 Ana Tedavi:\n• Maksimal güvenli rezeksiyon...",
      "tumor_location": "Tümör merkezi: (124, 89) piksel",
      "heatmap_filename": "heatmap_overlay.png",
      "original_filename": "original_image.png",
      "clinical_profile": { ... }
    }
    ```

### 3. Görsel Çıktı Servisleri
*   **Heatmap Görseli:** `GET /heatmaps/{filename}` (Grad-CAM ısı haritasını döndürür)
*   **Orijinal Görsel:** `GET /original/{filename}` (Sisteme yüklenen orijinal MR görselini döndürür)

---

## 🛡 6. İleri Düzey Sorun Giderme (Advanced Troubleshooting FAQ)

### Soru 1: Windows'ta `Scripts\activate : File ... cannot be loaded because running scripts is disabled` hatası alıyorum. Ne yapmalıyım?
*   **Cevap:** Bu hata, Windows PowerShell'in güvenlik nedeniyle dışarıdan betik çalıştırılmasını engellemesinden kaynaklanır. Bu engeli aşmak için sanal ortamı aktif etmeden doğrudan sunucuyu çağırabilirsiniz:
    ```powershell
    venv\Scripts\pip install -r requirements.txt
    $env:PYTHONPATH="backend"; venv\Scripts\python -m uvicorn backend.main:app --reload --port 8000
    ```
    Bu komutlar güvenlik politikasını tamamen bypass eder.

### Soru 2: `ModuleNotFoundError: No module named 'pandas'` veya `'xgboost'` hatası veriyor.
*   **Cevap:** Kütüphaneler sanal ortamınıza yüklenmemiş demektir. Proje dizinindeyken terminalde şu komutu çalıştırarak eksik kütüphaneleri anında yükleyin:
    ```powershell
    venv\Scripts\pip install pandas xgboost
    ```

### Soru 3: Sunucu başlarken `[ERROR] Model yuklenemedi` uyarısı verdi. Sistem çalışır mı?
*   **Cevap:** Evet, çalışır! Geliştirdiğimiz **akıllı fail-safe** mekanizması sayesinde, eğer `models/brain_mri_classifier.pth` dosyası bulunamazsa, sistem arka planda otomatik olarak geçici bir model oluşturup kaydeder. Sunucunuz çökmeksizin test modunda ayağa kalkar.

### Soru 4: `Address already in use` (Port 8000 dolu) hatası alıyorum.
*   **Cevap:** Bilgisayarınızda başka bir uygulama (örneğin eski bir sunucu veya docker konteyneri) `8000` portunu kullanıyor demektir. Sunucuyu farklı bir portta (örneğin `8080`) başlatmak için şu komutu kullanın:
    ```powershell
    $env:PYTHONPATH="backend"; venv\Scripts\python -m uvicorn backend.main:app --reload --port 8080
    ```

---

## ⚖️ Yasal Uyarı

Bu sistem tıbbi bir cihaz değildir. Teşhis ve tedavi önerileri yapay zeka tabanlı olup **demo ve eğitim amaçlıdır.** Gerçek sağlık süreçleriniz için lütfen uzman bir hekime danışınız.#   t e k n o f e s t  
 