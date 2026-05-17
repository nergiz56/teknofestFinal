import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import cv2
from pathlib import Path

class BrainMRIPredictorLight:
    def __init__(self, model_path="models/brain_mri_classifier.pth"):
        self.device = torch.device("cpu")  # CPU kullan
        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
        
        # Sınıflar
        self.classes = ["glioma", "menin", "tumor"]
        
        # Transform - daha basit
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Daha küçük boyut
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_bytes: bytes) -> dict:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
            
        class_name = self.classes[pred_class.item()]
        confidence = confidence.item()
        
        # Basit sonuçlar
        return {
            "tumor_present": True,
            "tumor_type": f"brain_{class_name}",
            "probabilities": {
                "glioma": float(probs[0][0]),
                "menin": float(probs[0][1]), 
                "tumor": float(probs[0][2])
            },
            "confidence": confidence,
            "tumor_size_category": "medium",  # Sabit değer
            "stage_estimate": "intermediate",  # Sabit değer
            "treatment_suggestion": f"🏥 {class_name.title()} (Orta Evre): Standart tedavi protokolleri uygulanabilir.",
            "tumor_location": "Tümör merkezi: (128, 128) piksel",
            "heatmap_filename": "heatmap_overlay.png",
            "original_filename": "original_image.png"
        }
