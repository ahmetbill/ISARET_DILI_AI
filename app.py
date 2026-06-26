import streamlit as st
import cv2
import difflib
from collections import deque
from ultralytics import YOLO
import os

# ==========================================
# 1. KERNEL: NLP VE KELİME TAMAMLAMA (Offline)
# ==========================================
class OfflineAutoCompleter:
    def __init__(self):
        self.vocabulary = [
            # --- Mevcut Listen ---
            "TEKNOLAB","MERHABA", "PROJE", "SİSTEM", "KONTROL", "ROBOT", "YAPAYZEKA",
            "OTOMASYON", "MÜHENDİS", "İTÜ", "BİLGİSAYAR", "GÖRÜNTÜ", "ASL",

            # --- En Sık Kullanılan Genel Kelimeler (Bağlaç, Edat, Zamir) ---
            "BİR", "VE", "BU", "İÇİN", "İLE", "DAHA", "ÇOK", "GİBİ", "KADAR",
            "SONRA", "KENDİ", "HER", "HANGİ", "TÜM", "BAŞKA", "AYNI", "ÇÜNKÜ",
            "ANCAK", "VEYA", "YADA", "EĞER", "SADECE", "YİNE", "HEM", "HİÇ",

            # --- Soru Kelimeleri ---
            "NE", "NASIL", "NEDEN", "NİÇİN", "KİM", "NEREDE", "NE ZAMAN", "HANGİSİ",

            # --- En Sık Kullanılan Günlük İsimler ve Sıfatlar ---
            "ŞEY", "ZAMAN", "GÜN", "YIL", "İNSAN", "İŞ", "HAYAT", "DÜNYA",
            "YENİ", "BÜYÜK", "İYİ", "İLK", "ÖNEMLİ", "SON", "GÜZEL", "KÜÇÜK",
            "YOL", "SONUÇ", "DURUM", "KONU", "BÖLÜM", "YER", "YÜKSEK", "HIZLI",

            # --- En Sık Kullanılan Fiiller (Mastar Formunda) ---
            "YAPMAK", "GELMEK", "GİTMEK", "ALMAK", "VERMEK",
            "BİLMEK", "BULMAK", "ÇIKMAK", "GÖRMEK", "İSTEMEK", "ÇALIŞMAK", 
            "DÜŞÜNMEK", "BAŞLAMAK", "GÖSTERMEK", "YAZMAK", "OKUMAK", "ANLAMAK",

            # --- Teknik, Geliştirme ve Proje Odaklı Sık Kullanılanlar ---
            "VERİ", "YAZILIM", "DONANIM", "SENSÖR", "MOTOR", "ALGORİTMA",
            "SİMÜLASYON", "İLETİŞİM", "PARAMETRE", "AĞ", "SUNUCU", "SÜRÜ",
            "HEDEF", "TAKİP", "TASARIM", "ANALİZ", "TEST", "KOD", "MODEL",
            "DÖNGÜ", "HATA", "ÇÖZÜM", "GÜNCELLEME", "KOMUT", "EKSEN", "BAĞLANTI"
        ]

    def suggest_word(self, partial_word, top_n=3):
        if not partial_word:
            return []
        partial_word = partial_word.upper()
        prefix_matches = [w for w in self.vocabulary if w.startswith(partial_word)]
        close_matches = difflib.get_close_matches(partial_word, self.vocabulary, n=top_n, cutoff=0.4)
        combined = list({w for w in (prefix_matches + close_matches)})
        return combined[:top_n]

# ==========================================
# 2. KERNEL: SİNYAL FİLTRELEME & BİRLEŞTİRİCİ
# ==========================================
class WordAssembler:
    VERSION = 2

    def __init__(self, debounce_frames=6):
        self.debounce_frames = debounce_frames
        self.frame_buffer = deque(maxlen=debounce_frames)
        self.current_word = ""
        self.full_sentence = ""
        self._v = self.VERSION

    def update(self, detected_char, suggestion=None):
        if not detected_char:
            return self.current_word, self.full_sentence

        self.frame_buffer.append(detected_char)

        if len(self.frame_buffer) == self.debounce_frames and len(set(self.frame_buffer)) == 1:
            val = self.frame_buffer[0]

            if val.upper() in ("SPACE", "YUMRUK", "FIST", "NOTHING", "DEL", "DELETE"):
                word_to_add = suggestion if suggestion else self.current_word
                if word_to_add:
                    self.full_sentence += word_to_add + " "
                    self.current_word = ""
                self.frame_buffer.clear()
            elif not self.current_word or self.current_word[-1] != val:
                self.current_word += val
                self.frame_buffer.clear()

        return self.current_word, self.full_sentence

# ==========================================
# 3. UI/UX: MINIMALIST MODERN PREMIUM DESIGN
# ==========================================

st.set_page_config(page_title="ASL Intelligence", layout="wide")

# Modern SaaS / Inter Font ve Soft Arayüz CSS enjeksiyonu
st.markdown("""
<style>
            
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]{
    font-family:'Inter',sans-serif !important;
    background:#f8fafc !important;
}

/* HERO */
.hero{
    background:linear-gradient(
        135deg,
        #0f172a 0%,
        #1e293b 100%
    );
    border-radius:24px;
    padding:60px;
    color:white;
    margin-bottom:30px;
}

.hero-title{
    font-size:3rem;
    font-weight:800;
    margin-bottom:10px;
    line-height:1.1;
}

.hero-sub{
    color:#cbd5e1;
    font-size:1.1rem;
    max-width:700px;
    line-height:1.6;
}

/* BADGE */
.status-badge{
    display:inline-block;
    background:rgba(255,255,255,0.1);
    backdrop-filter:blur(10px);
    padding:8px 16px;
    border-radius:999px;
    font-size:13px;
    font-weight:600;
    margin-bottom:20px;
}

/* FEATURE CARDS */
.feature-card{
    background:white;
    border:1px solid #e2e8f0;
    border-radius:18px;
    padding:24px;
    margin-bottom:20px;
    transition:all .25s ease;
}

.feature-card:hover{
    transform:translateY(-4px);
    box-shadow:0 12px 32px rgba(0,0,0,0.08);
}

.feature-card h4{
    margin-top:0;
    color:#0f172a;
}

.feature-card p{
    color:#64748b;
    margin-bottom:0;
}

/* MAIN PANELS */
.glass-panel{
    background:white;
    border:1px solid #e2e8f0;
    border-radius:20px;
    padding:24px;
    box-shadow:0 4px 20px rgba(0,0,0,0.04);
}

/* CAMERA AREA */
.camera-panel{
    background:white;
    border-radius:20px;
    border:1px solid #e2e8f0;
    padding:24px;
}

/* NLP OUTPUT */
.output-box{
    background:#f8fafc;
    border:1px solid #e2e8f0;
    border-radius:14px;
    padding:18px;
    margin-bottom:15px;
}

.output-label{
    color:#64748b;
    font-size:12px;
    font-weight:700;
    letter-spacing:0.08em;
}

.output-text{
    color:#0f172a;
    font-size:24px;
    font-weight:700;
}

/* WORD */
.current-word{
    color:#4338ca;
    font-size:28px;
    font-weight:700;
}

/* SUGGESTIONS */
.suggestion-chip{
    display:inline-block;
    background:#eef2ff;
    color:#4338ca;
    padding:8px 14px;
    border-radius:999px;
    font-size:13px;
    font-weight:600;
    margin-right:8px;
    margin-bottom:8px;
}

/* BUTTONS */
.stButton > button{
    width:100%;
    height:48px;
    border:none;
    border-radius:12px;
    background:#4338ca;
    color:white;
    font-weight:600;
}

.stButton > button:hover{
    background:#3730a3;
}

/* CHECKBOX */
.stCheckbox{
    padding-top:10px;
}

/* TABS */
.stTabs [data-baseweb="tab-list"]{
    gap:16px;
}

.stTabs [data-baseweb="tab"]{
    font-weight:600;
    font-size:15px;
    color:#64748b;
}

.stTabs [aria-selected="true"]{
    color:#4338ca !important;
}

/* EDUCATION GRID */
.letter-grid-item{
    background:white;
    border:1px solid #e2e8f0;
    border-radius:14px;
    padding:14px;
    text-align:center;
    transition:.2s;
}

.letter-grid-item:hover{
    transform:translateY(-3px);
    box-shadow:0 8px 20px rgba(0,0,0,.05);
}

/* STREAMLIT CONTAINER */
.block-container{
    padding-top:2rem;
    max-width:1400px;
}

/* IMAGE */
img{
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# Üst Bilgi Alanı (Minimal Header)
st.markdown("<h2 style='color: #0F172A; font-weight: 700; margin-bottom: 2px;'>ASL Translation Platform</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748B; font-size: 14px; margin-top: 0;'>TeknoLAB 2025-2026 Engine Core · Control & Automation Engineering</p>", unsafe_allow_html=True)
st.markdown("<div style='border-bottom: 1px solid #E2E8F0; margin-bottom: 24px;'></div>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Sistem Hatası: 'best.pt' yüklenemedi. Bilgi: {e}")
    model_loaded = False

if 'assembler' not in st.session_state or getattr(st.session_state.assembler, '_v', 0) != WordAssembler.VERSION:
    st.session_state.assembler = WordAssembler()
if 'completer' not in st.session_state:
    st.session_state.completer = OfflineAutoCompleter()

tab1, tab2 = st.tabs(["⚡ Canlı Çeviri Platformu", "📋 ASL & Özel Sınıf Kılavuzu"])

# ----------------- TAB 1: CANLI ÇEVİRİ -----------------
with tab1:
    col_cam, col_nlp = st.columns([1.8, 1.2], gap="large")

    with col_cam:
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; color:#334155; font-weight:600;'>Kamera Girişi</h4>", unsafe_allow_html=True)
        run_camera = st.checkbox("Görüntü İşleme Motorunu Aktif Et", value=False)
        FRAME_WINDOW = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_nlp:
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; color:#334155; font-weight:600;'>Doğal Dil İşleme (NLP)</h4>", unsafe_allow_html=True)
        
        # Temiz ve Modern Bilgi Alanları
        sentence_ph = st.empty()
        word_ph = st.empty()
        suggest_ph = st.empty()
        
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        if st.button("Arabelleği Temizle", width='stretch'):
            st.session_state.assembler.current_word = ""
            st.session_state.assembler.full_sentence = ""
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    def render_minimal_nlp(sentence, word, suggestions):
        sentence_ph.markdown(f"<small style='color:#64748B; font-weight:500;'>OLUŞAN CÜMLE</small><h2 style='color:#1E293B; font-weight:700; margin-top:4px; margin-bottom:16px;'>{sentence if sentence else '...'}</h2>", unsafe_allow_html=True)
        word_ph.markdown(f"<small style='color:#64748B; font-weight:500;'>MEVCUT KELİME</small><h4 style='color:#4F46E5; font-weight:600; margin-top:4px; margin-bottom:16px;'>{word if word else '...'}</h4>", unsafe_allow_html=True)
        
        sug_chips = "".join([f"<span style='background:#EEF2F6; color:#475569; padding:6px 12px; border-radius:6px; font-size:13px; font-weight:500; margin-right:6px;'>{s}</span>" for s in suggestions]) if suggestions else "<em style='color:#94A3B8; font-size:13px;'>Harf girişi bekleniyor...</em>"
        suggest_ph.markdown(f"<small style='color:#64748B; font-weight:500; display:block; margin-bottom:8px;'>AKILLI TAHMİNLER</small>{sug_chips}", unsafe_allow_html=True)

    if model_loaded and run_camera:
        cap = cv2.VideoCapture(0)
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Kamera akışı kesildi.")
                break

            results = model.predict(frame, conf=0.6, verbose=False)
            detected_char = None

            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                detected_char = model.names[int(box.cls)]

            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(annotated_frame, channels="RGB", width='stretch')

            suggs = st.session_state.completer.suggest_word(st.session_state.assembler.current_word)
            top_suggestion = suggs[0] if suggs else None
            cur_word, full_sent = st.session_state.assembler.update(detected_char, suggestion=top_suggestion)
            suggs = st.session_state.completer.suggest_word(cur_word)
            render_minimal_nlp(full_sent, cur_word, suggs)

        cap.release()
    else:
        if not run_camera:
            FRAME_WINDOW.image("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=800&auto=format&fit=crop", width='stretch', caption="Kamera Pasif durumdadır.")
        render_minimal_nlp(
            st.session_state.assembler.full_sentence,
            st.session_state.assembler.current_word,
            st.session_state.completer.suggest_word(st.session_state.assembler.current_word)
        )

# ----------------- TAB 2: EĞİTİM EKRANI -----------------
with tab2:
    import base64
    from PIL import Image as PILImage2
    import io as io2

    st.markdown("<h4 style='color:#334155;font-weight:600;margin-bottom:4px;'>ASL Alfabe Rehberi</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;font-size:14px;margin-bottom:20px;'>Amerikan İşaret Dili (ASL) alfabesindeki tüm harflerin el pozisyonları.</p>", unsafe_allow_html=True)

    def img_to_b64_sq(path, size=160):
        img = PILImage2.open(path).resize((size, size), PILImage2.LANCZOS)
        buf = io2.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    CARD = (
        "display:inline-block;"
        "background:white;"
        "border:1px solid #e2e8f0;"
        "border-radius:14px;"
        "overflow:hidden;"
        "text-align:center;"
        "vertical-align:top;"
        "width:calc(16.666% - 10px);"
        "margin:5px;"
    )
    LBL = "padding:8px 4px;font-weight:700;font-size:15px;color:#0f172a;display:block;"

    html = "<div style='width:100%;line-height:0;'>"
    for letter in [chr(i) for i in range(65, 91)]:
        img_path = f"assets/{letter}.png"
        if os.path.exists(img_path):
            b64 = img_to_b64_sq(img_path)
            html += (
                f'<div style="{CARD}">'
                f'<img src="data:image/png;base64,{b64}" style="width:100%;display:block;"/>'
                f'<span style="{LBL}">{letter}</span>'
                f'</div>'
            )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)