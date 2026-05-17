#!/usr/bin/env python3
"""
Railway Deploy Fix Script
Model boyutunu kucult ve memory limitini optimize et
"""

import torch
import os
from pathlib import Path

def optimize_model():
    """Model boyutunu kucult"""
    model_path = "models/brain_mri_classifier.pth"
    
    if not os.path.exists(model_path):
        print("❌ Model dosyasi bulunamadi!")
        return False
    
    try:
        # Modeli yukle
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Model state dict'i al
        if isinstance(checkpoint, dict):
            model_state = checkpoint
        else:
            model_state = checkpoint.state_dict()
        
        # Modeli quantize et (boyut kucultme)
        quantized_dict = torch.quantization.quantize_dynamic(
            model_state, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        
        # Quantize edilmis modeli kaydet
        quantized_path = "models/brain_mri_classifier_quantized.pth"
        torch.save(quantized_dict, quantized_path)
        
        print(f"✅ Model optimize edildi: {quantized_path}")
        print(f"📏 Orijinal boyut: {os.path.getsize(model_path)/1024/1024:.1f} MB")
        print(f"📏 Optimize boyut: {os.path.getsize(quantized_path)/1024/1024:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Model optimize hatasi: {e}")
        return False

if __name__ == "__main__":
    optimize_model()
