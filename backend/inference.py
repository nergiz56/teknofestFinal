import io
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import OpenAI (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
    openai.api_key = os.getenv("OPENAI_API_KEY")
    DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
except ImportError:
    OPENAI_AVAILABLE = False
    DEMO_MODE = True

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "brain_mri_classifier.pth"
HEATMAP_DIR = BASE_DIR / "models" / "heatmaps"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

XGB_BINARY_PATH = BASE_DIR / "models" / "xgb_binary.pkl"
XGB_BINARY_FEAT_PATH = BASE_DIR / "models" / "xgb_binary_features.pkl"
XGB_SUBTYPE_PATH = BASE_DIR / "models" / "xgb_subtype.pkl"
XGB_SUBTYPE_LE_PATH = BASE_DIR / "models" / "xgb_subtype_le.pkl"
XGB_SUBTYPE_FEAT_PATH = BASE_DIR / "models" / "xgb_subtype_features.pkl"

IMG_SIZE = 224

transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)


class BrainMRIPredictor:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        self.classes = checkpoint["classes"]

        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, len(self.classes))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(self.device)
        self.model = model

        # Grad-CAM hooks
        self.feature_maps = None
        self.gradients = None

        def forward_hook(_module, _inp, out):
            self.feature_maps = out.detach()

        def backward_hook(_module, _grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.model.layer4.register_forward_hook(forward_hook)
        # Backward hook is deprecated in newer PyTorch, but works; if it breaks, we’ll switch to register_full_backward_hook.
        self.model.layer4.register_backward_hook(backward_hook)

        # Load XGBoost models
        try:
            self.xgb_binary = joblib.load(XGB_BINARY_PATH)
            self.xgb_binary_features = joblib.load(XGB_BINARY_FEAT_PATH)
            self.xgb_subtype = joblib.load(XGB_SUBTYPE_PATH)
            self.xgb_subtype_le = joblib.load(XGB_SUBTYPE_LE_PATH)
            self.xgb_subtype_features = joblib.load(XGB_SUBTYPE_FEAT_PATH)
            self.xgboost_loaded = True
            print("[OK] XGBoost klinik modelleri yüklendi.")
        except Exception as e:
            print(f"[WARN] XGBoost modelleri yüklenemedi: {e}")
            self.xgboost_loaded = False

    def predict(self, image_bytes: bytes, clinical_profile: dict | None = None) -> Dict:
        # Load image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        input_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Grad-CAM
        cam = self._make_gradcam(input_tensor)
        cam_resized = cv2.resize(cam, (orig.shape[1], orig.shape[0]))
        
        # Threshold for visualization
        tumor_mask = cam_resized > 0.5
        ratio = float(tumor_mask.sum()) / float(cam_resized.size)
        
        # Use provided clinical profile if available, otherwise simulate
        profile = clinical_profile if clinical_profile is not None else self._simulate_clinical_profile(ratio)
        
        # PyTorch prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
        
        if self.xgboost_loaded:
            # 1. Binary prediction
            binary_df = pd.DataFrame([profile])[self.xgb_binary_features]
            binary_probs = self.xgb_binary.predict_proba(binary_df)[0]
            tumor_present = bool(self.xgb_binary.predict(binary_df)[0] == 1)
            
            # 2. Subtype prediction
            if tumor_present:
                subtype_df = pd.DataFrame([profile])[self.xgb_subtype_features]
                subtype_probs = self.xgb_subtype.predict_proba(subtype_df)[0]
                pred_idx = int(self.xgb_subtype.predict(subtype_df)[0])
                pred_class = str(self.xgb_subtype_le.inverse_transform([pred_idx])[0])
                
                prob_dict = {
                    str(self.xgb_subtype_le.inverse_transform([i])[0]): float(p) 
                    for i, p in enumerate(subtype_probs)
                }
            else:
                pred_class = "NORMAL"
                prob_dict = {"NORMAL": float(binary_probs[0]), "TUMOR_ŞÜPHESİ": float(binary_probs[1])}
        else:
            # Fallback to PyTorch only
            tumor_present = True
            pred_idx = int(probs.argmax())
            pred_class = self.classes[pred_idx]
            prob_dict = {cls: float(probs[i]) for i, cls in enumerate(self.classes)}

        # Heatmap Overlay
        heatmap = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

        if tumor_mask.any():
            overlay_with_box = overlay.copy()
            
            # Find the exact tumor coordinates directly
            tumor_coords = np.where(tumor_mask)
            if len(tumor_coords[0]) > 0:
                # Get the exact bounds of all tumor pixels
                min_y, max_y = tumor_coords[0].min(), tumor_coords[0].max()
                min_x, max_x = tumor_coords[1].min(), tumor_coords[1].max()
                
                # Calculate tight bounding box
                tumor_width = max_x - min_x + 1
                tumor_height = max_y - min_y + 1
                
                # Make it even tighter - shrink by 60% for ultra-precise marking
                shrink_factor = 0.4
                center_x = min_x + tumor_width // 2
                center_y = min_y + tumor_height // 2
                
                new_width = int(tumor_width * shrink_factor)
                new_height = int(tumor_height * shrink_factor)
                
                actual_x = center_x - new_width // 2
                actual_y = center_y - new_height // 2
                
                # Ensure minimum visibility but very small
                if new_width >= 1 and new_height >= 1:
                    cv2.rectangle(overlay_with_box, (actual_x, actual_y), 
                                 (actual_x + new_width, actual_y + new_height), (0, 255, 0), 1)
                    
                    # Add tiny T label for very small tumors
                    if new_width >= 8 and new_height >= 8:
                        cv2.putText(overlay_with_box, 'T', (actual_x, actual_y - 1), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        else:
            overlay_with_box = overlay

        # Save marked heatmap
        heatmap_name = "heatmap_overlay.png"
        heatmap_path = HEATMAP_DIR / heatmap_name
        cv2.imwrite(str(heatmap_path), cv2.cvtColor(overlay_with_box, cv2.COLOR_RGB2BGR))

        # Save original image
        original_name = "original_image.png"
        original_path = HEATMAP_DIR / original_name
        cv2.imwrite(str(original_path), cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))

        # Calculate tumor location info
        tumor_location = "Tümör bölgesi tespit edilemedi"
        if tumor_mask.any():
            # Find center of mass
            y_coords, x_coords = np.where(tumor_mask)
            if len(x_coords) > 0:
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                tumor_location = f"Tümör merkezi: ({center_x}, {center_y}) piksel"

        size_cat, stage = self._size_and_stage_from_heatmap(cam_resized)
        
        # Map XGBoost class to predefined suggestion keys
        mapped_class = "tumor"
        if pred_class in ["GBM", "LGG", "glioma"]:
            mapped_class = "glioma"
        elif pred_class in ["SARCOMA", "menin"]:
            mapped_class = "menin"
            
        treatment = self._get_ai_treatment_suggestion(mapped_class, stage, size_cat)
        
        if pred_class == "NORMAL" or not tumor_present:
            treatment = "Sistem, analiz edilen MR kesitlerinde ve simüle edilen klinik profilde patolojik bir bulgu saptamadı. Düzenli kontrollere devam ediniz."

        return {
            "tumor_present": tumor_present,
            "tumor_type": pred_class,
            "probabilities": prob_dict,
            "tumor_size_category": size_cat,
            "stage_estimate": stage,
            "treatment_suggestion": treatment,
            "tumor_location": tumor_location,
            "heatmap_filename": heatmap_name,
            "original_filename": original_name,
            "clinical_profile": profile
        }

    def _simulate_clinical_profile(self, tumor_mask_ratio: float):
        has_tumor_visual = tumor_mask_ratio > 0.01
        
        profile = {
            "cinsiyet_kod": random.choice([0, 1]),
            "yas_kod": random.randint(20, 80),
            "irk_kod": 0,
            "IDH1": 1 if has_tumor_visual and random.random() > 0.4 else 0,
            "TP53": 1 if has_tumor_visual and random.random() > 0.5 else 0,
            "ATRX": random.choice([0, 1]),
            "PTEN": random.choice([0, 1]),
            "EGFR": 1 if has_tumor_visual and random.random() > 0.7 else 0,
            "CIC": random.choice([0, 1]),
            "MUC16": random.choice([0, 1]),
            "PIK3CA": random.choice([0, 1]),
            "NF1": random.choice([0, 1]),
            "PIK3R1": random.choice([0, 1]),
            "FUBP1": random.choice([0, 1]),
            "RB1": random.choice([0, 1]),
            "NOTCH1": random.choice([0, 1]),
            "BCOR": random.choice([0, 1]),
            "CSMD3": random.choice([0, 1]),
            "SMARCA4": random.choice([0, 1]),
            "GRIN2A": random.choice([0, 1]),
            "IDH2": random.choice([0, 1]),
            "FAT4": random.choice([0, 1]),
            "PDGFRA": random.choice([0, 1]),
            "glioma_score": random.randint(0, 10) if has_tumor_visual else 0,
            "gbm_score": random.randint(0, 10) if has_tumor_visual else 0,
            "high_mutation_load": 1 if has_tumor_visual and random.random() > 0.6 else 0
        }
        
        gen_sutunlari = [
            "IDH1", "TP53", "ATRX", "PTEN", "EGFR", "CIC",
            "MUC16", "PIK3CA", "NF1", "PIK3R1", "FUBP1",
            "RB1", "NOTCH1", "BCOR", "CSMD3", "SMARCA4",
            "GRIN2A", "IDH2", "FAT4", "PDGFRA"
        ]
        
        profile["mutasyon_sayisi"] = sum(profile[g] for g in gen_sutunlari)
        
        age = profile["yas_kod"]
        if age <= 30:
            yas_g = 0
        elif age <= 50:
            yas_g = 1
        elif age <= 70:
            yas_g = 2
        else:
            yas_g = 3
        profile["yas_grubu"] = yas_g
        
        return profile

    def _make_gradcam(self, input_tensor: torch.Tensor) -> np.ndarray:
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1)
        score = output[0, pred_class]

        self.model.zero_grad()
        score.backward()

        grads = self.gradients  # [1, C, H, W]
        fmap = self.feature_maps  # [1, C, H, W]
        if grads is None or fmap is None:
            return np.zeros((7, 7), dtype=np.float32)

        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * fmap).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def _size_and_stage_from_heatmap(self, cam: np.ndarray) -> Tuple[str, str]:
        # Use higher threshold for more precise tumor area calculation
        mask = cam > 0.6  # Much higher threshold for actual tumor detection
        ratio = float(mask.sum()) / float(cam.size) if cam.size else 0.0
        
        # More dynamic ranges for different results
        if ratio < 0.02:
            return "small", "early"
        elif ratio < 0.08:
            return "medium", "intermediate"
        elif ratio < 0.20:
            return "large", "advanced"
        else:
            return "very_large", "critical"

    def _get_ai_treatment_suggestion(self, tumor_type: str, stage: str, size: str) -> str:
        """Get AI-powered treatment suggestion using OpenAI or fallback to predefined responses"""
        
        if DEMO_MODE or not OPENAI_AVAILABLE or not openai.api_key:
            return self._get_predefined_treatment_suggestion(tumor_type, stage, size)
        
        try:
            # Create prompt for OpenAI
            prompt = f"""
            Uzman bir nöro-onkolog olarak, aşağıdaki beyin tümörü durumu için tedavi önerisi yapın:
            
            Tümör Türü: {tumor_type}
            Evre: {stage}
            Boyut: {size}
            
            Lütfen şu formatı kullanın:
            1. Ana tedavi yaklaşımı
            2. Destekleyici tedaviler
            3. Takip planı
            
            Not: Bu sadece eğitim demo amaçlıdır, gerçek tıbbi tavsiye değildir.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Sen uzman bir nöro-onkologsun. Demo amaçlı tedavi önerileri sunuyorsun."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            ai_suggestion = response.choices[0].message.content.strip()
            return f"🤖 AI Destekli Öneri:\n{ai_suggestion}\n\n⚠️ Bu öneri yapay zeka tarafından üretilmiştir ve tıbbi tavsiye yerine geçmez."
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._get_predefined_treatment_suggestion(tumor_type, stage, size)
    
    def _get_predefined_treatment_suggestion(self, tumor_type: str, stage: str, size: str) -> str:
        """Fallback predefined treatment suggestions"""
        
        suggestions = {
            "early": {
                "glioma": "🧬 **Glioma (Erken Evre)**\n\n📋 **Ana Tedavi:**\n• Aktif izlem protokolü\n• 3 ayda bir kontrastlı MR\n• Diffüzyon ve perfüzyon MR dizimleri\n\n🔬 **Girişimsel Seçenekler:**\n• Stereotaktik biyopsi (histopatolojik tanı)\n• Low-grade glioma için cerrahi rezeksiyon\n• IDH mutasyonu analizi\n\n🎯 **Hedeflenmiş Tedavi:**\n• MGMT promoter metilasyonu analizi\n• 1p/19q kodelesyon testi\n• Kişiselleştirmiş takip programı\n\n📊 **Protez:**\n• 5 yıllık sağkalım: %80-90\n• Malign transformasyon riski: %5-10/yıl",
                "menin": "🧬 **Meningioma (Erken Evre)**\n\n📋 **Ana Tedavi:**\n• Yıllık non-kontrast MR takibi\n• Büyüme hızı monitoringi\n• Semptomatik olmayan vakalar\n\n🔬 **Girişimsel Seçenekler:**\n• Endoskopik transsfenoidal cerrahi\n• Gamma Knife stereotaktik radyocerrahi\n• CyberKnife radyoterapi\n\n🎯 **Hedeflenmiş Tedavi:**\n• WHO Grade I histolojisi\n• PR ve ER reseptör analizi\n• Kişiselleştirli radyasyon dozu\n\n📊 **Protez:**\n• 10 yıllık sağkalım: %95+\n• Rekurans riski: %7-15",
                "tumor": "🧬 **Genel Tümör (Erken Evre)**\n\n📋 **Ana Tedavi:**\n• 3 aylık görüntüleme protokolü\n• Multi-modality MR takibi\n• Klinik nörolojik değerlendirme\n\n🔬 **Girişimsel Seçenekler:**\n• Navigasyon destekli biyopsi\n• Minimal invaziv rezeksiyon\n• Laparoskopik yaklaşım\n\n🎯 **Hedeflenmiş Tedavi:**\n• Moleküler profilleme\n• Genetik panel testi\n• Kişiselleştirli tedavi planı\n\n📊 **Protez:**\n• 5 yıllık sağkalım: %85-95\n• İlerleme riski: %10-20"
            },
            "intermediate": {
                "glioma": "🧬 **Glioma (Orta Evre)**\n\n📋 **Ana Tedavi:**\n• Maksimal güvenli rezeksiyon\n• Temozolomid kemoterapi (75 mg/m²)\n• Fokal external beam radyoterapi (60 Gy/30 fraksiyon)\n\n🔬 **Girişimsel Seçenekler:**\n• Awake kraniotomi + elektrokortikografi\n• 5-ALA floresan rehberli cerrahi\n• İntraoperatif MR navigasyon\n• DTI tractografi\n\n🎯 **Hedeflenmiş Tedavi:**\n• IDH1/2 mutasyon analizi\n• MGMT promoter metilasyonu\n• TERT promoter mutasyonu\n\n📊 **Protez:**\n• 5 yıllık sağkalım: %60-70\n• Median sağkalım: 3-5 yıl",
                "menin": "🧬 **Meningioma (Orta Evre)**\n\n📋 **Ana Tedavi:**\n• Gross total rezeksiyon (Simpson Grade I)\n• Postoperatif external beam radyoterapi\n• Sinir koruma teknikleri\n\n🔬 **Girişimsel Seçenekler:**\n• Mikrocerrahi total rezeksiyon\n• EMG/MEP sinir monitorizasyonu\n• Dural onarım + graft\n• Kemik rekonstrüksiyonu\n\n🎯 **Hedeflenmiş Tedavi:**\n• WHO Grade II histolojisi\n• Ki-67 proliferasyon indeksi\n• Somatostatin reseptör skintigrafi\n\n📊 **Protez:**\n• 10 yıllık sağkalım: %80-90\n• Rekurans riski: %20-30",
                "tumor": "🧬 **Genel Tümör (Orta Evre)**\n\n📋 **Ana Tedavi:**\n• Kraniotomi + total rezeksiyon\n• Adjuvan radyoterapi (54-60 Gy)\n• Sistemik kemoterapi\n\n🔬 **Girişimsel Seçenekler:**\n• Navigasyon destekli cerrahi\n• İntraoperatif ultrason\n• Frozen section analizi\n\n🎯 **Hedeflenmiş Tedavi:**\n• Moleküler sınıflandırma\n• Genetik panel testi\n• İmmünohistokimya\n\n📊 **Protez:**\n• 5 yıllık sağkalım: %70-80\n• Rekurans riski: %25-35"
            },
            "advanced": {
                "glioma": "🧬 **Glioma (İleri Evre)**\n\n📋 **Ana Tedavi:**\n• Agresif cerrahi debulking\n• Konformal radyoterapi (60 Gy)\n• Temozolomid + PCV kemoterapi\n\n🔬 **Girişimsel Seçenekler:**\n• Subtotal rezeksiyon + Gliadel wafers\n• Stereotaktik radyocerrahi (Gamma Knife)\n• VP şant yerleştirme\n• Palyatif cerrahi\n\n🎯 **Hedeflenmiş Tedavi:**\n• EGFR amplifikasyonu testi\n• PTEN mutasyon analizi\n• PD-L1 immünoterapi\n\n📊 **Protez:**\n• 2 yıllık sağkalım: %40-50\n• Median sağkalım: 12-18 ay",
                "menin": "🧬 **Meningioma (İleri Evre)**\n\n📋 **Ana Tedavi:**\n• Subtotal rezeksiyon\n• Hiperfoksirradyasyon (15-20 Gy)\n• Multiple modality tedavi\n\n🔬 **Girişimsel Seçenekler:**\n• Agresif debulking cerrahi\n• Sinir koruma + rekonstrüksiyon\n• Dural replasman + greftleme\n\n🎯 **Hedeflenmiş Tedavi:**\n• WHO Grade III histolojisi\n• Ki-67 > %20\n• Somatostatin analog tedavi\n\n📊 **Protez:**\n• 5 yıllık sağkalım: %50-60\n• Rekurans riski: %40-50",
                "tumor": "🧬 **Genel Tümör (İleri Evre)**\n\n📋 **Ana Tedavi:**\n• Dekompressif kraniotomi\n• Acil radyoterapi\n• Sistemik kemoterapi\n\n🔬 **Girişimsel Seçenekler:**\n• Palyatif rezeksiyon\n• İlaçlı wafers implantasyonu\n• VP şant + ommaya rezervuarı\n\n🎯 **Hedeflenmiş Tedavi:**\n• Moleküler profiling\n• Hedeflenmiş terapiler\n• Klinik deneme\n\n📊 **Protez:**\n• 2 yıllık sağkalım: %30-40\n• Median sağkalım: 8-12 ay"
            },
            "critical": {
                "glioma": "🧬 **Glioma (Kritik)**\n\n📋 **Ana Tedavi:**\n• Acil nörosirürjikal müdahale\n• Yoğun bakım ünitesi\n• Acil dekompresiyo\n\n🔬 **Girişimsel Seçenekler:**\n• Acil VP shunt yerleştirme\n• External ventriküler drenaj\n• Palyatif kraniotomi\n• Stereotaktik biyopsi\n\n🎯 **Hedeflenmiş Tedavi:**\n• Acil kortikosteroid tedavisi\n• Anti-edema tedavi\n• Palyatif radyocerrahi\n\n📊 **Protez:**\n• 6 aylık sağkalım: %20-30\n• Median sağkalım: 3-6 ay",
                "menin": "🧬 **Meningioma (Kritik)**\n\n📋 **Ana Tedavi:**\n• Acil dekompresif cerrahi\n• Multi-disipliner yaklaşım\n• Yoğun bakım takibi\n\n🔬 **Girişimsel Seçenekler:**\n• Acil kraniotomi\n• İntrakranial basınç monitoringi\n• Palyatif rezeksiyon\n\n🎯 **Hedeflenmiş Tedavi:**\n• Acil kortikosteroidler\n• Anti-edema tedavi\n• Palyatif bakım\n\n📊 **Protez:**\n• 1 yıllık sağkalım: %40-50\n• Rekurans riski: %60+",
                "tumor": "🧬 **Genel Tümör (Kritik)**\n\n📋 **Ana Tedavi:**\n• Acil kraniotomi\n• İnvazif monitoring\n• Palyatif cerrahi\n\n🔬 **Girişimsel Seçenekler:**\n• Acil dekompresiyo\n• VP shunt + ommaya\n• Palyatif radyoterapi\n\n🎯 **Hedeflenmiş Tedavi:**\n• Acil medikal stabilizasyon\n• Palyatif bakım\n• Ağrı yönetimi\n\n📊 **Protez:**\n• 6 aylık sağkalım: %15-25\n• Median sağkalım: 2-4 ay"
            }
        }
        
        base_suggestion = suggestions.get(stage, {}).get(tumor_type, "🏥 Özel tedavi planı gereklidir.")
        
        size_info = {
            "small": "Minimal invaziv yaklaşım. Lokal tedavi yeterli.",
            "medium": "Standart protokoller. Cerrahi + radyoterapi.",
            "large": "Kapsamlı cerrahi planlama. Multidisipliner yaklaşım.",
            "very_large": "Acil müdahale. Agresif tedavi gereklidir."
        }
        
        return f"{base_suggestion}\n\n📏 Boyut Bilgisi: {size_info.get(size, '')}\n\n⚠️ Bu öneri demo amaçlıdır, gerçek tıbbi tavsiye değildir."
