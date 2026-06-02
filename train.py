"""Run this script to train the model before starting the app."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.ml_model import train

if __name__ == "__main__":
    print("Starting model training...")
    metrics = train()
    print("\n=== Training Complete ===")
    print(f"ROC-AUC:           {metrics['roc_auc']}")
    print(f"Avg Precision:     {metrics['avg_precision']}")
    print(f"Brier Score:       {metrics['brier_score']}")
    print("\nClassification Report:")
    print(metrics["classification_report"])
