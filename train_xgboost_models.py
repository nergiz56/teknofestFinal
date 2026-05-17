"""
XGBoost Model Egitim Scripti
Binary (tumor var/yok) + Subtype (GBM/LGG/SARCOMA/TUMOR) modelleri egitir
ve models/ klasorune kaydeder.
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BINARY_DATA = r"C:\Users\Behzat\Downloads\VeriSeti\Final\DATASETS\KANSER_MASTER_FEATURE_V4.xlsx"
SUBTYPE_DATA = r"C:\Users\Behzat\Downloads\VeriSeti\Final\DATASETS\SUBTYPE_FEATURE_V2.xlsx"

# Feature kolumları (kullanici formuyla eslesen)
FEATURE_COLS = [
    "cinsiyet_kod", "yas_kod", "irk_kod",
    "IDH1", "TP53", "ATRX", "PTEN", "EGFR", "CIC",
    "MUC16", "PIK3CA", "NF1", "PIK3R1", "FUBP1",
    "RB1", "NOTCH1", "BCOR", "CSMD3", "SMARCA4",
    "GRIN2A", "IDH2", "FAT4", "PDGFRA",
    "mutasyon_sayisi", "yas_grubu",
    "glioma_score", "gbm_score", "high_mutation_load"
]

SUBTYPE_FEATURE_COLS = [
    "cinsiyet_kod", "yas_kod", "irk_kod",
    "IDH1", "TP53", "ATRX", "PTEN", "EGFR", "CIC",
    "MUC16", "PIK3CA", "NF1", "PIK3R1", "FUBP1",
    "RB1", "NOTCH1", "BCOR", "CSMD3", "SMARCA4",
    "GRIN2A", "IDH2", "FAT4", "PDGFRA",
    "mutasyon_sayisi", "yas_grubu"
]


def train_binary():
    print("\n[1/2] Binary model egitiliyor (Tumor var/yok)...")
    df = pd.read_excel(BINARY_DATA)
    print(f"  Veri: {df.shape[0]} kayit, {df.shape[1]} sutun")
    print(f"  Sinif dagilimi: {dict(df['tumor_binary'].value_counts())}")

    # Sadece mevcut feature kolonlarini kullan
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available]
    y = df["tumor_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(classification_report(y_test, y_pred, target_names=["NO_TUMOR", "TUMOR"]))

    # Kaydet
    joblib.dump(model, MODELS_DIR / "xgb_binary.pkl")
    joblib.dump(available, MODELS_DIR / "xgb_binary_features.pkl")
    print(f"  Kaydedildi: models/xgb_binary.pkl")
    return acc


def train_subtype():
    print("\n[2/2] Subtype model egitiliyor (GBM/LGG/SARCOMA/TUMOR)...")
    df = pd.read_excel(SUBTYPE_DATA)

    # Sadece tumorlu vakalari al
    df = df[df["grade"] != "NO_TUMOR"].copy()
    print(f"  Veri: {df.shape[0]} kayit (tumorlu)")
    print(f"  Sinif dagilimi: {dict(df['grade'].value_counts())}")

    # Feature engineering
    gen_sutunlari = [
        "IDH1", "TP53", "ATRX", "PTEN", "EGFR", "CIC",
        "MUC16", "PIK3CA", "NF1", "PIK3R1", "FUBP1",
        "RB1", "NOTCH1", "BCOR", "CSMD3", "SMARCA4",
        "GRIN2A", "IDH2", "FAT4", "PDGFRA"
    ]
    available_gen = [c for c in gen_sutunlari if c in df.columns]
    df["mutasyon_sayisi"] = df[available_gen].sum(axis=1)
    df["yas_grubu"] = pd.cut(
        df["yas_kod"], bins=[0, 30, 50, 70, 150], labels=[0, 1, 2, 3]
    ).astype(int)

    available = [c for c in SUBTYPE_FEATURE_COLS if c in df.columns]
    X = df[available]
    y = df["grade"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"  Sinif kodlari: {dict(zip(le.classes_, range(len(le.classes_))))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softmax",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Kaydet
    joblib.dump(model, MODELS_DIR / "xgb_subtype.pkl")
    joblib.dump(le, MODELS_DIR / "xgb_subtype_le.pkl")
    joblib.dump(available, MODELS_DIR / "xgb_subtype_features.pkl")
    print(f"  Kaydedildi: models/xgb_subtype.pkl")
    return acc


if __name__ == "__main__":
    print("=" * 50)
    print("XGBoost Model Egitimi Basliyor...")
    print("=" * 50)

    binary_acc = train_binary()
    subtype_acc = train_subtype()

    print("\n" + "=" * 50)
    print("EGITIM TAMAMLANDI!")
    print(f"  Binary  accuracy : {binary_acc*100:.1f}%")
    print(f"  Subtype accuracy : {subtype_acc*100:.1f}%")
    print("=" * 50)
    print("\nOlusturulan dosyalar:")
    print("  models/xgb_binary.pkl          - Binary tumor tespiti")
    print("  models/xgb_binary_features.pkl - Binary feature listesi")
    print("  models/xgb_subtype.pkl         - Subtype siniflandirici")
    print("  models/xgb_subtype_le.pkl      - Subtype label encoder")
    print("  models/xgb_subtype_features.pkl- Subtype feature listesi")
