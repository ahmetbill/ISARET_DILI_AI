# 🤟 TeknoLAB ASL AI - Gerçek Zamanlı İşaret Dili Tanıma Sistemi

Bu proje, **TeknoGenç TeknoLAB 2025-2026 Dönem Sonu Projesi** kapsamında geliştirilmiş, uçtan uca çalışan bir bilgisayarlı görü (Computer Vision) ve doğal dil işleme (NLP) ürünüdür. 

Sistem, Amerikan İşaret Dili (ASL) alfabesindeki harfleri ve özel komutları anlık olarak tanıyıp anlamlı kelimelere ve cümlelere dönüştürür.

## 🎯 Projenin Amacı ve Başarı Kriterleri
* **Temel İşlev:** İşitme engelli bireyler ile iletişim bariyerlerini yıkmak, işaret dilini çevrimdışı ve yüksek hızda metne dönüştürmek.
* **Doğruluk Oranı (Accuracy):** $\ge \%85$ hedefi, uygulanan veri artırımı (augmentation) teknikleriyle başarıyla yakalanmıştır.
* **Tepki Süresi (Response Time):** Tek aşamalı YOLO11 Nano mimarisi sayesinde $\le 0.5$ saniye seviyesine indirilmiştir (Yönerge hedefi $\le 2$ sn).

## 🏛️ Kültürel Farkındalık ve Vizyon (ASL & TİD)
İşaret dilleri evrensel değildir; her kültürün kendi sözdizimi ve grameri bulunur. Literatürde açık kaynaklı büyük verilerin **Amerikan İşaret Dili (ASL)** üzerinde yoğunlaşması nedeniyle, bu prototip sistemin model omurgasını ASL alfabesiyle eğittik. 

Ancak **TeknoLAB** ekibi olarak kurduğumuz bu esnek *YOLO11 + Çevrimdışı NLP* mimarisi, yakın gelecekte iki el koordinasyonuna dayalı olan **Türk İşaret Dili (TİD)** veri setleriyle beslenerek yerli ve milli bir erişilebilirlik çözümüne dönüştürülecek şekilde tasarlanmıştır.

## 🛠️ Teknik Kurallar ve Kısıtlamalar (Offline Mimari)
* **API Yasağı:** Projede hiçbir ticari bulut API'si (ChatGPT, Google Vision vb.) kullanılmamıştır. Tüm çıkarım (inference) ve kelime tahmin süreçleri lokal ortamda döner.
* **Offline Çalışma:** İnternet bağlantısı tamamen kesildiğinde bile sistem aynı hız ve performansla çalışmaya devam eder.

## 📦 Kurulum ve Çalıştırma

1.  **Depoyu Klonlayın:**
    ```bash
    git clone [https://github.com/KULLANICI_ADIN/PROJE_REPO_ADIN.git](https://github.com/KULLANICI_ADIN/PROJE_REPO_ADIN.git)
    cd PROJE_REPO_ADIN
    ```

2.  **Sanal Ortam Oluşturun ve Aktif Edin:**
    ```bash
    python -m venv myenv
    # Windows için:
    myenv\Scripts\activate
    # Linux/Mac için:
    source myenv/bin/activate
    ```

3.  **Gerekli Bağımlılıkları Kurun:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Uygulamayı Başlatın:**
    `best.pt` model ağırlık dosyasını kök dizine ekledikten sonra şu komutu çalıştırın:
    ```bash
    streamlit run app.py
    ```

## ⚙️ Model ve Eğitim Detayları
* **Mimari:** YOLO11 Nano (Transfer Learning / Fine-Tuning)
* **Epoch / Batch Size:** 50 Epoch / 16 Batch
* **Veri Artırımı (Data Augmentation):** Modelin ezberlemesini (overfitting) önlemek amacıyla Roboflow üzerinden *Cutout (Random Erasing)*, *Rotation ($\pm15^\circ$)*, *Brightness ($\pm15\%$)* ve *Horizontal Flip* teknikleri uygulanmıştır. Bu sayede model eksik, yarım veya ters açılı elleri bile başarıyla tanır.
* **Sinyal Filtreleme (Debounce):** Kamera akışındaki anlık titremeleri engellemek için 5 frame'lik bir kuyruk yapısı (deque) kullanılarak stabilite sağlanmıştır.

## 👥 Takım Rolleri
* **Veri Sorumlusu:** TeknoLAB Ekip Üyesi 1 (Veri toplama, Roboflow Augmentation yönetimi)
* **Modelleme Sorumlusu:** TeknoLAB Ekip Üyesi 2 (Google Colab T4 GPU Eğitim süreçleri ve Fine-Tuning)
* **Arayüz & Entegrasyon Sorumlusu:** TeknoLAB Ekip Üyesi 3 (Streamlit geliştirme ve Çevrimdışı NLP Entegrasyonu)