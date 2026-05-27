import cv2
import os

# --- AYARLAR ---
video_yolu = 'yumruk_videosu.mp4' # Videonun tam adı veya yolu
cikis_klasoru = 'dataset_yumruk'  # Fotoğrafların kaydedileceği klasör
kac_karede_bir = 10               # Her 10 karede 1 fotoğraf kaydeder (Çok benzer fotoğrafları önlemek için)

# Çıkış klasörü yoksa otomatik oluştur
if not os.path.exists(cikis_klasoru):
    os.makedirs(cikis_klasoru)

# Videoyu yükle
cap = cv2.VideoCapture(video_yolu)

sayac = 0
kayit_sayaci = 0

print("Video işleniyor, lütfen bekleyin...")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break # Video bittiğinde döngüden çık

    # Sadece belirlediğimiz aralıklarla kaydet (Örn: Her 10 karede bir)
    if sayac % kac_karede_bir == 0:
        dosya_adi = os.path.join(cikis_klasoru, f"yumruk_acili_{kayit_sayaci}.jpg")
        cv2.imwrite(dosya_adi, frame)
        kayit_sayaci += 1
        
    sayac += 1

# İşlem bitince belleği temizle
cap.release()
print(f"İşlem tamam! Toplam {kayit_sayaci} adet fotoğraf '{cikis_klasoru}' klasörüne kaydedildi.")