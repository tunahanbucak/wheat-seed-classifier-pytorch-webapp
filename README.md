# 🌾 WheatVision AI — Buğday Tohumu Sınıflandırma Servisi

Uçtan uca PyTorch yapay sinir ağı (MLP) ile eğitilmiş, 7 geometrik/morfolojik tohum parametresinden buğday varyetesini (**Kama**, **Rosa**, **Canadian**) tespit eden yüksek performanslı **FastAPI** web servisi.

---

## 🚀 Mimari & Özellikler

* **Derin Öğrenme Modeli:** PyTorch `nn.Module` (7 Girdi -> Linear(16) -> ReLU -> Linear(16) -> ReLU -> 3 Çıktı).
* **Test Doğruluğu:** `%85.71` (Stratified Test Seti).
* **Ön İşleme:** Eğitim setinde fit edilmiş `StandardScaler` nesnesi ile veri sızıntısız çıkarım.
* **Backend:** FastAPI + Pydantic veri doğrulaması + Asenkron endpoint mimarisi.
* **Frontend:** Modern koyu mod, interaktif kaydırıcılar (sliders), anlık Softmax olasılık göstergeleri ve hızlı test butonları.

---

## 🛠️ Kurulum & Çalıştırma

1. Sanal ortamı aktifleştirin ve gereksinimleri kurun:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Geliştirici sunucusunu başlatın:
```bash
uvicorn main:app --reload --port 8001
```

3. Tarayıcınızda açın:
* **Web Arayüzü:** [http://localhost:8001](http://localhost:8001)
* **Otomatik API Dokümantasyonu (Swagger):** [http://localhost:8001/docs](http://localhost:8001/docs)

---

## 📡 API Kullanımı (cURL Örneği)

```bash
curl -X POST "http://localhost:8001/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "area": 14.88,
    "perimeter": 14.57,
    "compactness": 0.881,
    "length_of_kernel": 5.554,
    "width_of_kernel": 3.333,
    "asymmetry_coefficient": 1.018,
    "length_of_groove": 4.956
  }'
```

### Örnek Yanıt:
```json
{
  "success": true,
  "data": {
    "variety": "Kama",
    "confidence_percentage": 99.97,
    "probabilities": {
      "Kama": 99.97,
      "Rosa": 0.02,
      "Canadian": 0.01
    },
    "title": "Kama Buğdayı (Triticum aestivum)",
    "origin": "Polonya / Orta Avrupa",
    "description": "Orta boyutta ve yüksek adaptasyon yeteneğine sahip yaygın kışlık buğday varyetesi."
  }
}
```

---

Geliştirici: **Tunahan Buçak** • Full-Stack AI Engineer
