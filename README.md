# TeknoLAB ASL Tanıma Sistemi

**Ekip:** Anthropic
**Demo:** [huggingface.co/spaces/isiq/asl-intelligence](https://huggingface.co/spaces/isiq/asl-intelligence)
**GitHub:** [github.com/ahmetbill/ISARET_DILI_AI](https://github.com/ahmetbill/ISARET_DILI_AI)

TeknoGenç TeknoLAB 2025-2026 Dönem Sonu Projesi kapsamında geliştirilmiş, uçtan uca çalışan bir bilgisayarlı görü (Computer Vision) ve doğal dil işleme (NLP) sistemidir. Sistem, Amerikan İşaret Dili (ASL) alfabesindeki harfleri gerçek zamanlı olarak tanıyıp anlamlı kelimelere ve cümlelere dönüştürür.

---

## Projenin Amacı

İşitme engelli bireyler günlük iletişimlerinde işaret dilini kullanmakta; ancak karşı taraftaki bireylerin büyük çoğunluğu bu dili bilmemektedir. Bu proje, söz konusu iletişim bariyerini ortadan kaldırmak amacıyla yalnızca standart bir web kamerası kullanarak ASL alfabesini gerçek zamanlı tanıyan, tamamen çevrimdışı çalışan bir yapay zeka sistemi geliştirmeyi hedeflemektedir.

### Başarı Kriterleri

| Kriter | Hedef | Gerçekleşen |
|--------|-------|-------------|
| mAP50 (Doğruluk) | >= %85 | %96.8 |
| Yanıt Süresi | <= 2 sn | <= 0.5 sn |
| Offline Çalışma | Zorunlu | Tam uyum |

---

## Teknik Mimari

### Model
- **Mimari:** YOLO11 Nano (Fine-Tuning / Transfer Learning)
- **Eğitim:** 50 Epoch, Batch Size 16, AdamW optimizer
- **Veri Seti:** Roboflow — American Sign Language Letters v3 (~7.833 görüntü, 29 sınıf)
- **Veri Artırımı:** Cutout, Rotation (±15°), Brightness (±15%), Horizontal Flip
- **GPU:** Google Colab — NVIDIA T4

### Sınıflar
A'dan Z'ye 26 harf + özel `YUMRUK` sınıfı (kelime tamamlama sinyali olarak kullanılır).

### Uygulama Katmanları

| Katman | Bileşen | Teknoloji |
|--------|---------|-----------|
| Giriş | Web kamerası | OpenCV |
| Model | YOLO11n (best.pt) | Ultralytics / PyTorch |
| İş Mantığı | WordAssembler + öneri kabulü | Python (deque) |
| NLP | OfflineAutoCompleter | difflib |
| Arayüz | Web UI | Streamlit |
| Dağıtım | Bulut hosting | Hugging Face Spaces |

### Offline Çalışabilirlik
Tüm model ağırlıkları ve otomatik tamamlama sözlüğü yerel ortamda çalışmaktadır. İnternet bağlantısı yalnızca ilk kurulum için gereklidir.

---

## Kurulum ve Çalıştırma

**1. Depoyu klonlayın:**
```bash
git clone https://github.com/ahmetbill/ISARET_DILI_AI.git
cd ISARET_DILI_AI
```

**2. Sanal ortam oluşturun:**
```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# Linux / Mac
source myenv/bin/activate
```

**3. Bağımlılıkları kurun:**
```bash
pip install -r requirements.txt
```

**4. Uygulamayı başlatın:**
```bash
python -m streamlit run app.py
```

Tarayıcıda `http://localhost:8501` adresine gidin.

> Kurulum yapmadan kullanmak için: [huggingface.co/spaces/isiq/asl-intelligence](https://huggingface.co/spaces/isiq/asl-intelligence)

---

## Kullanım

1. "Görüntü İşleme Motorunu Aktif Et" kutusunu işaretleyin.
2. Elinizi kameraya göstererek ASL harfleri oluşturun.
3. Tanınan harfler sağ panelde birleşerek kelime oluşturur.
4. Yumruk hareketi yapıldığında sistem önerilen kelimeyi cümleye ekler.
5. "Arabelleği Temizle" butonu ile sıfırlayabilirsiniz.

---

## Proje Yapısı

```
ISARET_DILI_AI/
├── app.py                  # Streamlit uygulaması
├── best.pt                 # Eğitilmiş YOLO11n model ağırlıkları
├── ASL_Proje.ipynb         # Eğitim notebook'u (tüm epoch çıktıları)
├── videodan_foto.py        # Veri toplama scripti
├── requirements.txt        # Bağımlılıklar
├── assets/                 # ASL alfabe görselleri (A-Z)
└── README.md
```

---

## Performans Metrikleri

| Epoch | mAP50 | Precision | Recall | mAP50-95 |
|-------|-------|-----------|--------|----------|
| 1 | 0.150 | 0.380 | 0.150 | 0.088 |
| 10 | 0.700 | 0.740 | 0.780 | 0.480 |
| 20 | 0.951 | 0.880 | 0.919 | 0.628 |
| 30 | 0.973 | 0.941 | 0.948 | 0.642 |
| 50 | **0.968** | **0.963** | **0.952** | **0.640** |

---

## Ekip ve Roller

| İsim | Roller |
|------|--------|
| Ahmet Bill | Veri Sorumlusu / Modelleme Sorumlusu / Arayüz & Entegrasyon Sorumlusu |

---

## Kaynakça

- Jocher, G. et al. (2024). Ultralytics YOLO11. https://github.com/ultralytics/ultralytics
- Roboflow. (2024). American Sign Language Letters Dataset (v3). https://roboflow.com
- Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.
- Streamlit Inc. (2024). Streamlit. https://streamlit.io
