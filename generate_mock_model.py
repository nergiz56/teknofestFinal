import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

def main():
    print("Mock model generation started...")
    
    # Define directories
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "brain_mri_classifier.pth"
    
    # Classes expected by inference.py
    classes = ["glioma", "menin", "tumor"]
    num_classes = len(classes)
    
    # Build ResNet18 model with 3 output classes
    print("Creating ResNet18 model architecture...")
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # Save mock state_dict
    print("Saving mock state_dict...")
    checkpoint = {
        "classes": classes,
        "model_state_dict": model.state_dict()
    }
    
    torch.save(checkpoint, model_path)
    print(f"Mock model successfully saved to: {model_path}")
    print("You can now start the FastAPI server and make real predictions!")

if __name__ == "__main__":
    main()
