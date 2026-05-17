#!/usr/bin/env python3
"""
Railway Deploy Fix Script
Model boyutunu küçült ve memory limitini optimize et
"""

import torch
import os
from pathlib import Path

def optimize_model():
    """Model boyutunu küçült"""
    model_path = "models/brain_mri_classifier.pth"
    
    if not os.path.exists(model_path):
        print("❌ Model dosyası bulunamadı!")
        return False
    
    try:
        # Modeli yükle
        model = torch.load(model_path, map_location='cpu')
        
        # Modeli quantize et (boyut küçültme)
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        
        # Quantize edilmiş modeli kaydet
        quantized_path = "models/brain_mri_classifier_quantized.pth"
        torch.save(quantized_model, quantized_path)
        
        print(f"✅ Model optimize edildi: {quantized_path}")
        print(f"📏 Orijinal boyut: {os.path.getsize(model_path)/1024/1024:.1f} MB")
        print(f"📏 Optimize boyut: {os.path.getsize(quantized_path)/1024/1024:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Model optimize hatası: {e}")
        return False

if __name__ == "__main__":
    optimize_model()
