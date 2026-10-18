import os
import torch
from torch import nn
import joblib
import numpy as np

# Variety descriptions and facts
VARIETY_METADATA = {
    "Kama": {
        "title": "Kama Buğdayı (Triticum aestivum)",
        "origin": "Polonya / Orta Avrupa",
        "description": "Orta boyutta ve yüksek adaptasyon yeteneğine sahip yaygın kışlık buğday varyetesi. Ekmeklik un üretiminde dengeli gluten yapısıyla tercih edilir.",
        "badge_color": "#10b981", # Emerald
    },
    "Rosa": {
        "title": "Rosa Buğdayı (Triticum durum)",
        "origin": "Doğu Avrupa / Akdeniz Havzası",
        "description": "Geniş ve ağır taneleriyle karakterize edilen yüksek verimli makarnalık ve sert buğday çeşidi. Yüksek protein oranı ve sarı pigment zenginliğiyle bilinir.",
        "badge_color": "#f59e0b", # Amber
    },
    "Canadian": {
        "title": "Canadian Buğdayı (Canadian Western Red Spring)",
        "origin": "Kanada Ovaları / Kuzey Amerika",
        "description": "Kuzey iklimine dayanıklı, sert kırmızı ilkbahar buğdayı. Dünyanın en kaliteli ekmeklik buğdaylarından biri olarak kabul edilir; çok yüksek gluten ve fermantasyon gücüne sahiptir.",
        "badge_color": "#ef4444", # Red
    }
}

class WheatSeedsClassifier(nn.Module):
    """
    Buğday Tohumu Sınıflandırma Yapay Sinir Ağı Mimarisi (PyTorch MLP)
    Input (7) -> Linear(16) -> ReLU -> Linear(16) -> ReLU -> Linear(3)
    """
    def __init__(self, in_features: int = 7, out_features: int = 3, hidden_units: int = 16):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


class WheatModelService:
    def __init__(self, model_dir: str = "models"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.base_dir, model_dir)
        
        # Paths
        self.weights_path = os.path.join(self.model_dir, "wheat_seeds_model.pth")
        self.scaler_path = os.path.join(self.model_dir, "wheat_scaler.joblib")
        self.labels_path = os.path.join(self.model_dir, "wheat_labels.joblib")
        
        # Load artifacts
        self.scaler = joblib.load(self.scaler_path)
        self.labels = joblib.load(self.labels_path)
        
        # Load model
        self.device = torch.device("cpu")
        self.model = WheatSeedsClassifier(in_features=7, out_features=3, hidden_units=16)
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.eval()

    def predict(self, features: list[float]) -> dict:
        """
        features: [area, perimeter, compactness, length_of_kernel, width_of_kernel, asymmetry_coefficient, length_of_groove]
        """
        # 1. Scale inputs using the training scaler
        arr = np.array(features, dtype=np.float32).reshape(1, -1)
        scaled_arr = self.scaler.transform(arr)
        
        # 2. Convert to PyTorch Tensor
        tensor_input = torch.tensor(scaled_arr, dtype=torch.float32).to(self.device)
        
        # 3. Model Inference (no gradient calculation)
        with torch.no_grad():
            logits = self.model(tensor_input)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = torch.argmax(logits, dim=1).item()
            
        variety_name = self.labels[pred_idx]
        confidence = float(probs[pred_idx])
        
        prob_dict = {
            self.labels[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(self.labels))
        }
        
        meta = VARIETY_METADATA.get(variety_name, {})
        
        return {
            "variety": variety_name,
            "confidence_percentage": round(confidence * 100, 2),
            "probabilities": prob_dict,
            "title": meta.get("title", variety_name),
            "origin": meta.get("origin", "Bilinmiyor"),
            "description": meta.get("description", ""),
            "badge_color": meta.get("badge_color", "#10b981")
        }

# Singleton instance
wheat_service = WheatModelService()
