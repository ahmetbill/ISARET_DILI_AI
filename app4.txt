import streamlit as st
import cv2
import difflib
from collections import deque
from ultralytics import YOLO

# ==========================================
# 1. NLP VE KELİME TAMAMLAMA MODÜLÜ (Offline)
# ==========================================
class OfflineAutoCompleter:
    def __init__(self):
        self.vocabulary = [
            "MERHABA", "PROJE", "SİSTEM", "KONTROL", "ROBOT", "YAPAYZEKA",
            "OTOMASYON", "MÜHENDİS", "DENEME", "BİLGİSAYAR", "GÖRÜNTÜ",
            "HELLO", "WORLD", "SYSTEM", "CONTROL", "ROBOTICS", "TEKNOLAB",
            "AUTOMATION", "ENGINEER", "PROJECT", "COMPUTER", "VISION"
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
# 2. SİNYAL FİLTRELEME & AKILLI "SPACE" MANTIĞI
# ==========================================
class WordAssembler:
    def __init__(self, debounce_frames=5):
        self.debounce_frames = debounce_frames
        self.frame_buffer = deque(maxlen=debounce_frames)
        self.current_word = ""
        self.full_sentence = ""

    def update(self, detected_char):
        if not detected_char:
            return self.current_word, self.full_sentence

        self.frame_buffer.append(detected_char)

        if len(self.frame_buffer) == self.debounce_frames and len(set(self.frame_buffer)) == 1:
            val = self.frame_buffer[0]

            if val == "SPACE" or val == "YUMRUK":
                if self.current_word:
                    self.full_sentence += self.current_word + " "
                    self.current_word = ""
                self.frame_buffer.clear()

            elif not self.current_word or self.current_word[-1] != val:
                self.current_word += val
                self.frame_buffer.clear()

        return self.current_word, self.full_sentence


# ==========================================
# 3. SAYFA AYARLARI
# ==========================================
st.set_page_config(page_title="TeknoLAB ASL AI", layout="wide", page_icon="🤟")

# ==========================================
# 4. MASTER CSS — GRAIN + DARK/WHITE BOUNDED
# ==========================================
st.markdown("""
<style>
/* ── GOOGLE FONTS ── */
@import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,300;1,400&display=swap');

/* ── CSS VARIABLES ── */
:root {
    --bg:           #0F0F0F;
    --bg-2:         #161616;
    --bg-3:         #1E1E1E;
    --surface:      #242424;
    --surface-2:    #2C2C2C;
    --border:       rgba(255,255,255,0.08);
    --border-strong:rgba(255,255,255,0.14);
    --text-primary: #F0EDE8;
    --text-secondary:#9A9591;
    --text-muted:   #5C5956;
    --accent:       #E8DDD0;
    --accent-warm:  #D4A574;
    --accent-cool:  #8EB4C8;
    --white-card:   #F7F4F0;
    --white-card-2: #EDE9E3;
    --white-text:   #1A1714;
    --white-sub:    #6B6560;
    --radius-sm:    10px;
    --radius-md:    16px;
    --radius-lg:    22px;
    --radius-xl:    28px;
    --shadow-sm:    0 1px 4px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.04);
    --shadow-md:    0 4px 20px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.06);
    --shadow-lg:    0 12px 48px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.07);
}

/* ── GLOBAL ── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: var(--bg) !important;
    font-family: 'Poppins', sans-serif !important;
    color: var(--text-primary) !important;
}

/* SVG noise grain overlay on entire app */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    opacity: 0.028;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    background-size: 200px 200px;
}

.block-container {
    padding: 2.5rem 2.5rem 5rem !important;
    max-width: 1440px !important;
    position: relative;
    z-index: 1;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.4rem !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Poppins', sans-serif !important;
}

.sb-logo {
    display: flex;
    align-items: center;
    gap: 11px;
    padding-bottom: 22px;
    margin-bottom: 22px;
    border-bottom: 1px solid var(--border);
}
.sb-logo-mark {
    width: 38px; height: 38px;
    border-radius: var(--radius-sm);
    background: var(--white-card);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    box-shadow: var(--shadow-sm);
    flex-shrink: 0;
}
.sb-logo-name {
    font-size: 13px; font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.2px;
}
.sb-logo-sub {
    font-size: 10px; font-weight: 400;
    color: var(--text-muted);
    letter-spacing: 0.5px;
    margin-top: 1px;
}

.sb-group-label {
    font-size: 9px; font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 20px 0 8px;
}

.sb-status {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 12px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    background: var(--surface);
    margin-top: 6px;
}
.sb-status-dot {
    width: 6px; height: 6px; border-radius: 50%; flex-shrink:0;
    animation: blink 2.4s ease infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
.sb-status-dot.on  { background: #6DBF7E; }
.sb-status-dot.off { background: #E06060; }
.sb-status-text {
    font-size: 11px; font-weight: 600;
    color: var(--text-secondary);
}

.sb-info-box {
    margin-top: 16px;
    padding: 12px 14px;
    border-radius: var(--radius-sm);
    background: var(--surface);
    border: 1px solid var(--border);
}
.sb-info-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0;
}
.sb-info-row + .sb-info-row {
    border-top: 1px solid var(--border);
}
.sb-info-key {
    font-size: 10px; font-weight: 500; color: var(--text-muted);
    letter-spacing: 0.3px;
}
.sb-info-val {
    font-size: 11px; font-weight: 700; color: var(--accent-warm);
}

/* ── SLIDER OVERRIDE ── */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent-warm) !important;
}
[data-testid="stSlider"] label,
[data-testid="stSlider"] p {
    font-family: 'Poppins', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
}

/* ── CHECKBOX ── */
[data-testid="stCheckbox"] label {
    font-family: 'Poppins', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
}

/* ── BUTTON ── */
[data-testid="stButton"] > button {
    font-family: 'Poppins', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    background: var(--surface-2) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-sm) !important;
    padding: 9px 18px !important;
    transition: all 0.18s ease !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stButton"] > button:hover {
    background: var(--white-card) !important;
    color: var(--white-text) !important;
    border-color: transparent !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.5) !important;
}

/* ── TABS ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: var(--radius-sm) !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 2px !important;
    width: fit-content;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    padding: 8px 22px !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    border: none !important;
    letter-spacing: 0.2px !important;
    transition: all 0.18s ease !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--white-card) !important;
    color: var(--white-text) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
}
[data-testid="stTabContent"] { padding-top: 30px !important; }

/* ── PAGE HEADER ── */
.page-hero {
    margin-bottom: 36px;
}
.page-hero-kicker {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 10px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    color: var(--accent-warm);
    margin-bottom: 12px;
}
.page-hero-kicker::before {
    content: '';
    width: 18px; height: 1px;
    background: var(--accent-warm);
    display: inline-block;
}
.page-hero-title {
    font-size: 42px; font-weight: 800;
    color: var(--text-primary);
    line-height: 1.08;
    letter-spacing: -1px;
    margin-bottom: 10px;
}
.page-hero-title em {
    font-style: normal;
    color: var(--text-muted);
    font-weight: 300;
}
.page-hero-desc {
    font-size: 14px; font-weight: 400;
    color: var(--text-secondary);
    max-width: 480px;
    line-height: 1.65;
}

/* ── SECTION LABEL ── */
.sec-label {
    font-size: 9px; font-weight: 700;
    letter-spacing: 1.8px; text-transform: uppercase;
    color: var(--text-muted);
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 14px;
}
.sec-label::after {
    content:''; flex:1; height:1px;
    background: var(--border);
}

/* ── CAMERA CARD (DARK) ── */
.cam-card {
    background: var(--bg-2);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-strong);
    box-shadow: var(--shadow-lg);
    overflow: hidden;
    position: relative;
}
/* grain on cam card */
.cam-card::after {
    content:'';
    position:absolute; inset:0;
    pointer-events:none;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='g'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23g)'/%3E%3C/svg%3E");
    background-size: 128px 128px;
    opacity: 0.04;
    border-radius: var(--radius-lg);
    z-index: 10;
}
.cam-topbar {
    display: flex; align-items: center; gap: 7px;
    padding: 13px 16px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-3);
}
.cam-dot { width: 11px; height: 11px; border-radius: 50%; }
.cam-dot-r { background: #FF5F57; }
.cam-dot-y { background: #FEBC2E; }
.cam-dot-g { background: #28C840; }
.cam-window-title {
    font-size: 11px; font-weight: 600;
    color: var(--text-muted);
    margin-left: 5px; letter-spacing: 0.3px;
}
.cam-live-pill {
    margin-left: auto;
    display: flex; align-items: center; gap: 5px;
    background: rgba(255,80,80,0.12);
    border: 1px solid rgba(255,80,80,0.2);
    color: #FF6060;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 10px; font-weight: 700; letter-spacing: 1px;
}
.cam-live-dot {
    width: 5px; height: 5px;
    background: #FF6060; border-radius: 50%;
    animation: blink 1s infinite;
}
.cam-body { padding: 14px; }

.cam-offline {
    aspect-ratio: 16/9;
    background: var(--bg-3);
    border-radius: var(--radius-md);
    border: 1px solid var(--border);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 10px;
}
.cam-offline-icon { font-size: 36px; opacity: 0.18; }
.cam-offline-text {
    font-size: 12px; font-weight: 500;
    color: var(--text-muted);
}

/* ── WHITE METRIC CARDS ── */
.metric-stack { display: flex; flex-direction: column; gap: 12px; }

.metric-card-white {
    background: var(--white-card);
    border-radius: var(--radius-md);
    padding: 18px 20px 16px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 2px 12px rgba(0,0,0,0.35), 0 0 0 1px rgba(255,255,255,0.6) inset;
    position: relative;
    overflow: hidden;
}
/* grain on white cards */
.metric-card-white::before {
    content:'';
    position:absolute; inset:0;
    pointer-events:none;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='g'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23g)'/%3E%3C/svg%3E");
    background-size: 128px 128px;
    opacity: 0.025;
    border-radius: var(--radius-md);
}
.metric-card-white.dark-var {
    background: var(--surface);
    border-color: var(--border-strong);
    box-shadow: var(--shadow-md);
}
.mc-label {
    font-size: 9px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: var(--white-sub);
    margin-bottom: 8px;
    display: flex; align-items: center; gap: 6px;
}
.metric-card-white.dark-var .mc-label { color: var(--text-muted); }
.mc-value {
    font-size: 22px; font-weight: 700;
    color: var(--white-text);
    line-height: 1.2;
    min-height: 28px;
    letter-spacing: -0.3px;
    word-break: break-all;
}
.metric-card-white.dark-var .mc-value { color: var(--text-primary); }
.mc-value.empty {
    font-size: 14px; font-weight: 400;
    color: var(--white-sub);
    font-style: italic;
}
.metric-card-white.dark-var .mc-value.empty { color: var(--text-muted); }

/* chips */
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 2px; }
.chip {
    background: var(--white-card-2);
    color: var(--white-text);
    border: 1px solid rgba(0,0,0,0.1);
    border-radius: 6px;
    padding: 4px 11px;
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.5px;
    font-family: 'Poppins', sans-serif;
}
.chip-none { font-size: 12px; color: var(--white-sub); font-style: italic; }

/* stats mini row */
.stats-mini { display: flex; gap: 8px; margin-top: 10px; }
.stat-mini-card {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 6px;
    text-align: center;
}
.stat-mini-val {
    font-size: 18px; font-weight: 800;
    color: var(--accent-warm);
    line-height: 1;
}
.stat-mini-lbl {
    font-size: 9px; font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-top: 4px;
}

/* ── IMAGE OVERRIDE ── */
[data-testid="stImage"] img {
    border-radius: var(--radius-md) !important;
}

/* ── LEARNING TAB ── */
.guide-head {
    font-size: 24px; font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.4px; margin-bottom: 4px;
}
.guide-sub {
    font-size: 13px; font-weight: 400;
    color: var(--text-secondary);
    margin-bottom: 22px; line-height: 1.6;
}

/* sign card — WHITE surface */
.sign-card {
    background: var(--white-card);
    border-radius: var(--radius-lg);
    padding: 22px 20px 18px;
    border: 1px solid rgba(0,0,0,0.07);
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    position: relative; overflow: hidden;
    height: 100%;
}
.sign-card::after {
    content:'';
    position:absolute; inset:0;
    pointer-events:none;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='g'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.88' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23g)'/%3E%3C/svg%3E");
    background-size: 128px 128px;
    opacity: 0.025;
    border-radius: var(--radius-lg);
}
.sign-big-letter {
    font-size: 60px; font-weight: 900;
    color: var(--white-text);
    line-height: 1; margin-bottom: 10px;
    letter-spacing: -2px;
}
.sign-big-letter.sp {
    font-size: 26px;
    color: var(--white-sub);
    letter-spacing: -0.5px;
}
.sign-name-tag {
    font-size: 9px; font-weight: 700;
    letter-spacing: 1.8px; text-transform: uppercase;
    color: var(--white-sub);
    margin-bottom: 8px;
}
.sign-body {
    font-size: 12px; font-weight: 400;
    color: var(--white-sub);
    line-height: 1.65;
}
.sign-badge {
    display: inline-block;
    margin-top: 14px;
    padding: 3px 10px;
    border-radius: 5px;
    font-size: 9px; font-weight: 800;
    letter-spacing: 1px; text-transform: uppercase;
    background: rgba(0,0,0,0.07);
    color: var(--white-sub);
}
.sign-badge.special {
    background: rgba(212,165,116,0.15);
    color: #9A6A3A;
}

/* practice cam */
.practice-wrap {
    background: var(--bg-2);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-strong);
    box-shadow: var(--shadow-lg);
    overflow: hidden;
}
.practice-topbar {
    display: flex; align-items: center; gap: 8px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-3);
}
.practice-title {
    font-size: 10px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: var(--text-muted);
}
.practice-body { padding: 14px; }

/* detected label */
.detect-box {
    margin-top: 12px;
    padding: 14px 16px;
    border-radius: var(--radius-md);
    background: var(--surface);
    border: 1px solid var(--border);
    text-align: center;
}
.detect-label {
    font-size: 9px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: var(--text-muted); margin-bottom: 6px;
}
.detect-val {
    font-size: 32px; font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.5px;
    line-height: 1;
}
.detect-val.empty { color: var(--text-muted); font-size: 22px; font-weight: 300; }

/* info strip */
.info-strip {
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    display: flex; gap: 14px; align-items: flex-start;
    margin-bottom: 24px;
}
.info-strip-bar {
    width: 3px; border-radius: 3px;
    background: var(--accent-warm);
    align-self: stretch; flex-shrink: 0;
    min-height: 40px;
}
.info-strip-text {
    font-size: 12px; color: var(--text-secondary);
    line-height: 1.65; font-weight: 400;
}
.info-strip-text strong { color: var(--accent-warm); font-weight: 700; }

/* spec pills */
.spec-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 20px; }
.spec-pill {
    background: var(--white-card);
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 2px 10px rgba(0,0,0,0.35);
    flex: 1; min-width: 100px;
    position: relative; overflow: hidden;
}
.spec-pill::after {
    content:'';
    position:absolute; inset:0;
    pointer-events:none;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='g'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23g)'/%3E%3C/svg%3E");
    background-size: 128px 128px;
    opacity: 0.025;
}
.spec-val {
    font-size: 22px; font-weight: 800;
    color: var(--white-text);
    letter-spacing: -0.5px; line-height: 1;
    margin-bottom: 4px;
}
.spec-lbl {
    font-size: 9px; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase;
    color: var(--white-sub);
}

/* hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* error/warning override */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
}

</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-mark">🤟</div>
        <div>
            <div class="sb-logo-name">TeknoLAB ASL</div>
            <div class="sb-logo-sub">2025–2026 · AI Vision System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-group-label">Model</div>', unsafe_allow_html=True)

    @st.cache_resource
    def load_local_model():
        return YOLO("best.pt")

    try:
        model = load_local_model()
        st.markdown("""
        <div class="sb-status">
            <div class="sb-status-dot on"></div>
            <span class="sb-status-text">Lokal Model Aktif</span>
        </div>
        """, unsafe_allow_html=True)
        model_ok = True
    except Exception:
        st.markdown("""
        <div class="sb-status">
            <div class="sb-status-dot off"></div>
            <span class="sb-status-text">best.pt bulunamadı</span>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
        model_ok = False

    st.markdown('<div class="sb-group-label">Parametreler</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Güven Eşiği", 0.30, 1.00, 0.60)
    debounce_frames = st.slider("Debounce Frame", 3, 15, 5)

    st.markdown("""
    <div class="sb-info-box">
        <div class="sb-info-row">
            <span class="sb-info-key">Mimari</span>
            <span class="sb-info-val">YOLO11</span>
        </div>
        <div class="sb-info-row">
            <span class="sb-info-key">İşleme</span>
            <span class="sb-info-val">OpenCV</span>
        </div>
        <div class="sb-info-row">
            <span class="sb-info-key">NLP</span>
            <span class="sb-info-val">Offline</span>
        </div>
        <div class="sb-info-row">
            <span class="sb-info-key">Bulut API</span>
            <span class="sb-info-val">Sıfır</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── SESSION STATE ──────────────────────────────────────────────────────────
if 'assembler' not in st.session_state:
    st.session_state.assembler = WordAssembler(debounce_frames=debounce_frames)
if 'completer' not in st.session_state:
    st.session_state.completer = OfflineAutoCompleter()


# ── PAGE HERO ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
    <div class="page-hero-kicker">TeknoLAB · YOLO11 + OpenCV</div>
    <div class="page-hero-title">İşaret Dili <em>Tanıma</em><br>Sistemi</div>
    <div class="page-hero-desc">Gerçek zamanlı ASL algılama, çevrimdışı NLP ve akıllı kelime tamamlama.</div>
</div>
""", unsafe_allow_html=True)


# ── TABS ───────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["  🎥  Canlı Çeviri  ", "  📚  ASL Öğrenme Kılavuzu  "])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — CANLI ÇEVİRİ
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    col_cam, col_nlp = st.columns([3, 2], gap="large")

    # ── KAMERA SÜTUNU ────────────────────────────────────────────────────
    with col_cam:
        st.markdown('<div class="sec-label">Kamera Akışı</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="cam-card">
            <div class="cam-topbar">
                <div class="cam-dot cam-dot-r"></div>
                <div class="cam-dot cam-dot-y"></div>
                <div class="cam-dot cam-dot-g"></div>
                <span class="cam-window-title">ASL Vision · YOLO11</span>
                <div class="cam-live-pill">
                    <div class="cam-live-dot"></div>LIVE
                </div>
            </div>
            <div class="cam-body">
        """, unsafe_allow_html=True)

        run_camera = st.checkbox("Kamerayı Aktif Et", value=False)
        FRAME_WINDOW = st.empty()

        if not run_camera:
            st.markdown("""
            <div class="cam-offline">
                <div class="cam-offline-icon">📷</div>
                <div class="cam-offline-text">Aktif etmek için kutuyu işaretleyin</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    # ── NLP SÜTUNU ───────────────────────────────────────────────────────
    with col_nlp:
        st.markdown('<div class="sec-label">NLP Çıktıları</div>', unsafe_allow_html=True)

        sentence_ph = st.empty()
        word_ph = st.empty()
        suggest_ph = st.empty()

        def render_nlp(full_sentence, current_word, suggestions):
            # Cümle
            s_val   = full_sentence.strip()
            s_html  = f'<div class="mc-value">{s_val}</div>' if s_val else '<div class="mc-value empty">Henüz cümle oluşmadı…</div>'
            # Kelime
            w_html  = f'<div class="mc-value">{current_word}</div>' if current_word else '<div class="mc-value empty">Bekleniyor…</div>'
            # Tahminler
            if suggestions:
                chips = "".join([f'<div class="chip">{s}</div>' for s in suggestions])
                sg_html = f'<div class="chips">{chips}</div>'
            else:
                sg_html = '<div class="chip-none">Harf girişi bekleniyor…</div>'

            sentence_ph.markdown(f"""
            <div class="metric-card-white" style="margin-bottom:12px;">
                <div class="mc-label">📋 &nbsp; Oluşan Cümle</div>
                {s_html}
            </div>""", unsafe_allow_html=True)

            word_ph.markdown(f"""
            <div class="metric-card-white" style="margin-bottom:12px;">
                <div class="mc-label">🔤 &nbsp; Mevcut Kelime</div>
                {w_html}
            </div>""", unsafe_allow_html=True)

            suggest_ph.markdown(f"""
            <div class="metric-card-white dark-var">
                <div class="mc-label">💡 &nbsp; Akıllı Tahminler</div>
                {sg_html}
            </div>""", unsafe_allow_html=True)

        render_nlp(
            st.session_state.assembler.full_sentence,
            st.session_state.assembler.current_word,
            st.session_state.completer.suggest_word(st.session_state.assembler.current_word)
        )

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        if st.button("🧹  Ekranı Temizle", use_container_width=True):
            st.session_state.assembler.current_word = ""
            st.session_state.assembler.full_sentence = ""
            st.rerun()

        char_c = len(st.session_state.assembler.full_sentence.replace(" ", ""))
        word_c = len([w for w in st.session_state.assembler.full_sentence.split() if w])
        cam_icon = "●" if run_camera else "○"

        st.markdown(f"""
        <div class="stats-mini">
            <div class="stat-mini-card">
                <div class="stat-mini-val">{char_c}</div>
                <div class="stat-mini-lbl">Karakter</div>
            </div>
            <div class="stat-mini-card">
                <div class="stat-mini-val">{word_c}</div>
                <div class="stat-mini-lbl">Kelime</div>
            </div>
            <div class="stat-mini-card">
                <div class="stat-mini-val" style="color:{'#6DBF7E' if run_camera else 'var(--text-muted)'}; font-size:22px;">{cam_icon}</div>
                <div class="stat-mini-lbl">Kamera</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── KAMERA DÖNGÜSÜ (mantık korundu) ─────────────────────────────────
    if run_camera:
        cap = cv2.VideoCapture(0)
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Kameraya erişilemedi!")
                break

            results = model.predict(frame, conf=conf_threshold, verbose=False)
            detected_char = None

            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                detected_char = model.names[int(box.cls)]

            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(annotated_frame, use_container_width=True)

            current_word, full_sentence = st.session_state.assembler.update(detected_char)
            suggestions = st.session_state.completer.suggest_word(current_word)
            render_nlp(full_sentence, current_word, suggestions)

        cap.release()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — ASL ÖĞRENME KILAVUZU VE HİKAYE
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    
    st.markdown("""
    <div style="background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 24px; margin-bottom: 20px; position: relative; overflow: hidden; box-shadow: var(--shadow-sm);">
        <h3 style="color: var(--text-primary); margin-top: 0; font-size: 20px; font-weight: 700; letter-spacing: -0.5px;">📌 ASL ve Türk İşaret Dili (TİD)</h3>
        <p style="color: var(--text-secondary); font-size: 14px; line-height: 1.7; margin-bottom: 0;">
        Sanılanın aksine işaret dilleri evrensel değildir; her ülkenin kendi kültürüyle yoğrulmuş bağımsız bir dili, grameri ve sözdizimi vardır. Örneğin <strong>Amerikan İşaret Dili (ASL)</strong> alfabe harflerini ağırlıklı olarak tek elle ifade ederken, <strong>Türk İşaret Dili (TİD)</strong> harfler için genellikle iki elin koordinasyonunu kullanır. İşaret dilleri, sadece ellerin değil; yüz mimiklerinin ve vücut duruşunun kullanıldığı zengin ve yaşayan dillerdir.
        </p>
    </div>
    """, unsafe_allow_html=True)

    hist_col1, hist_col2 = st.columns(2, gap="large")
    with hist_col1:
        st.markdown("""
        <div style="background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 20px; height: 100%; box-shadow: var(--shadow-sm);">
            <h4 style="color: var(--accent-warm); margin-top: 0; font-size: 16px; font-weight: 700;">🏛️ Kültürel Kökenler</h4>
            <p style="color: var(--text-secondary); font-size: 13px; line-height: 1.6; margin-bottom: 0;">
            ASL'nin kökleri 19. yüzyılın başlarına uzanır ve Fransız İşaret Dili (LSF) ile harmanlanmıştır. Türk İşaret Dili (TİD) ise Osmanlı dönemindeki saray kültürüne ve sağırlar okullarına kadar uzanan çok derin ve eşsiz bir tarihe sahiptir. Her iki dil de sağır toplumunun en büyük kültürel mirasıdır.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with hist_col2:
        st.markdown("""
        <div style="background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 20px; height: 100%; box-shadow: var(--shadow-sm);">
            <h4 style="color: var(--accent-warm); margin-top: 0; font-size: 16px; font-weight: 700;">🚀 Neden ASL?</h4>
            <p style="color: var(--text-secondary); font-size: 13px; line-height: 1.6; margin-bottom: 0;">
            Yapay zeka modellerini eğitmek için devasa ve etiketlenmiş veri setlerine ihtiyaç vardır. Altyapımızı test etmek için açık kaynak verisi en yaygın olan ASL'yi seçtik. Ancak TeknoLAB olarak asıl hedefimiz; kurduğumuz bu çevrimdışı <em>YOLO11 + NLP</em> motorunu yakın gelecekte <strong>TİD</strong> veri setleriyle eğitip ülkemizdeki işitme engelli bireylerin hayatına dokunmaktır.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── MEVCUT KILAVUZ (INFO STRIP) ───────────────────────────────────────
    st.markdown("""
    <div class="info-strip">
        <div class="info-strip-bar"></div>
        <div class="info-strip-text">
            Sistemi kullanırken harf harf yazmanın yorucu olabileceğini biliyoruz. Bu yüzden sistemimiz <strong>NLP Kelime Tahmin Motoru</strong> ile donatılmıştır. Sadece birkaç harf girin ve <strong>YUMRUK (SPACE)</strong> komutuyla kelimeyi otomatik onaylayın! </n> Zaten sanılanın aksine ASL ve diğer işaret dilleri kelimeleri sadece nadir durumlarda harf harf yazar.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_guide, col_practice = st.columns([3, 2], gap="large")

    # ── KILAVUZ SÜTUNU ───────────────────────────────────────────────────
    with col_guide:
        st.markdown('<div class="sec-label">Desteklenen İşaretler</div>', unsafe_allow_html=True)

        signs = [
            {
                "letter": "A", "is_sp": False,
                "name": "",
                "desc": "Yumruk kapalı tutulur, başparmak dışarıda yanda konumlanır. El sıkıca yumruk yapmış gibi görünür.",
                "badge": "Temel Harf", "special": False,
            },
            {
                "letter": "B", "is_sp": False,
                "name": "",
                "desc": "Dört parmak açık, düz ve yapışık tutulur. Başparmak içe, avuç içe doğru katlanır.",
                "badge": "Temel Harf", "special": False,
            },
            {
                "letter": "C", "is_sp": False,
                "name": "",
                "desc": "El yarı açık tutularak büyük 'C' harfi şeklinde kıvrılır. Tüm parmaklar eşit şekilde bükülür.",
                "badge": "Temel Harf", "special": False,
            },
            {
                "letter": "YUMRUK", "is_sp": True,
                "name": "SPACE Komutu",
                "desc": "Tüm parmaklar sıkıca kapanır. Sistem bunu kelime bitişi ve boşluk sinyali olarak algılar.",
                "badge": "Özel Komut", "special": True,
            },
        ]

        gc1, gc2 = st.columns(2, gap="medium")
        cols = [gc1, gc2]
        for i, sign in enumerate(signs):
            with cols[i % 2]:
                letter_cls = "sign-big-letter sp" if sign["is_sp"] else "sign-big-letter"
                badge_cls  = "sign-badge special" if sign["special"] else "sign-badge"
                st.markdown(f"""
                <div class="sign-card">
                    <div class="{letter_cls}">{sign['letter']}</div>
                    <div class="sign-name-tag">{sign['name']}</div>
                    <div class="sign-body">{sign['desc']}</div>
                    <div class="{badge_cls}">{sign['badge']}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        st.markdown('<div class="sec-label" style="margin-top:8px;">Sistem Özellikleri</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="spec-row">
            <div class="spec-pill">
                <div class="spec-val">≤0.5s</div>
                <div class="spec-lbl">Tepki Süresi</div>
            </div>
            <div class="spec-pill">
                <div class="spec-val">YOLO11</div>
                <div class="spec-lbl">Mimari</div>
            </div>
            <div class="spec-pill">
                <div class="spec-val">%0</div>
                <div class="spec-lbl">Bulut API</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── PRATİK SÜTUNU ────────────────────────────────────────────────────
    with col_practice:
        st.markdown('<div class="sec-label">Pratik Ekranı</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="practice-wrap">
            <div class="practice-topbar">
                <span style="font-size:12px;">📷</span>
                <span class="practice-title">Canlı Test Kamerası</span>
            </div>
            <div class="practice-body">
        """, unsafe_allow_html=True)

        run_practice = st.checkbox("Pratik Kamerasını Aç", value=False, key="practice_cam")
        PRACTICE_WINDOW = st.empty()

        if not run_practice:
            st.markdown("""
            <div class="cam-offline" style="aspect-ratio:4/3;">
                <div class="cam-offline-icon">🤟</div>
                <div class="cam-offline-text">İşaretleri test etmek için aktif edin</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        detect_ph = st.empty()
        detect_ph.markdown("""
        <div class="detect-box">
            <div class="detect-label">Algılanan İşaret</div>
            <div class="detect-val empty">—</div>
        </div>
        """, unsafe_allow_html=True)

        if run_practice:
            cap2 = cv2.VideoCapture(0)
            while run_practice:
                ret, frame = cap2.read()
                if not ret:
                    break

                results2 = model.predict(frame, conf=conf_threshold, verbose=False)
                detected = None

                if len(results2[0].boxes) > 0:
                    b2 = results2[0].boxes[0]
                    detected = model.names[int(b2.cls)]

                ann = results2[0].plot()
                ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                PRACTICE_WINDOW.image(ann, use_container_width=True)

                if detected:
                    detect_ph.markdown(f"""
                    <div class="detect-box" style="border-color:var(--accent-warm);">
                        <div class="detect-label" style="color:var(--accent-warm);">Algılanan İşaret</div>
                        <div class="detect-val">{detected}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    detect_ph.markdown("""
                    <div class="detect-box">
                        <div class="detect-label">Algılanan İşaret</div>
                        <div class="detect-val empty">—</div>
                    </div>
                    """, unsafe_allow_html=True)

            cap2.release()