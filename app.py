import streamlit as st
import time
import random
from PIL import Image
import io
import base64
import os
import numpy as np
import torch
import torch.nn as nn
import requests

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

from torchvision import transforms
MODEL_NAME = "model.fix.pth"

MODEL_URL = (
    "https://github.com/Hafiizalhaq/"
    "deteksidaunmangga/releases/download/"
    "v.1/model.fix.pth"
)


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deteksi Penyakit Daun Mangga",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- PYTORCH MODEL LOADER ---

@st.cache_resource
def load_prediction_model(model_path):
    device = torch.device("cpu")
    model = resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048,512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512,128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128,4)
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Preprocessing utility

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor

# --- ROBUST CUSTOM CSS INJECTION (FORCED THEME) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* FORCE GLOBAL COLORS TO PREVENT THEME CONFLICTS */
    :root {
        --primary-color: #16a34a;
        --text-color: #064e3b;
        --background-color: #fcfdfb;
    }
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(135deg, #f0fdf4 0%, #f9fef8 100%) !important;
        color: #064e3b !important;
    }

    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #dcfce7 !important;
    }

    /* Override system styles enforcing text contrast */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, span {
        color: #064e3b !important;
    }

    /* Fix Native Streamlit Containers to look like our Custom Cards */
    [data-testid="stVerticalBlockBorderWrapper"] > div:has([data-testid="element-container"]) {
        background: white !important;
        border-radius: 1.25rem !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 15px rgba(6, 78, 59, 0.03), 0 2px 5px rgba(0,0,0,0.01) !important;
        border: 1px solid #eefdf2 !important;
    }

    /* Header Aesthetics */
    .fancy-header {
        background: white;
        border-radius: 1rem;
        border: 1px solid #dcfce7;
        padding: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        margin-bottom: 1.5rem;
    }
    .header-icon-box {
        width: 3.5rem; height: 3.5rem;
        background: linear-gradient(135deg, #16a34a, #059669);
        border-radius: 0.85rem;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 8px 16px -4px rgba(22, 163, 74, 0.4);
        flex-shrink: 0;
    }
    .header-title-text {
        font-weight: 800; font-size: 1.8rem;
        margin: 0; line-height: 1.1;
        color: #064e3b !important;
    }
    .header-subtitle-text {
        font-size: 0.9rem; margin: 0.3rem 0 0 0;
        color: #16a34a !important; font-weight: 500;
    }

    /* Uploader Box Styling override */
    [data-testid="stFileUploader"] section {
        background-color: #fcfdfb !important;
        border: 2px dashed #86efac !important;
        border-radius: 0.75rem !important;
        padding: 1rem !important;
    }
    
    /* Result Box inside predictions */
    .fancy-result-banner {
        padding: 1rem 1.25rem;
        background: linear-gradient(to right, #16a34a, #059669);
        color: white !important;
        border-radius: 0.75rem;
        font-weight: 700; font-size: 1.2rem;
        margin-top: 0.75rem;
        box-shadow: 0 2px 8px rgba(22, 163, 74, 0.2);
    }
    .fancy-result-banner p, .fancy-result-banner span {
        color: white !important;
    }

    /* Custom Components for layout clean-up */
    .item-card-inner {
        display: flex;
        gap: 1rem; padding: 1.1rem;
        background-color: #f9fef9;
        border: 1px solid #ecfdf5;
        border-radius: 0.85rem;
        margin-bottom: 0.85rem;
        transition: all 0.2s ease;
    }
    .item-card-inner:hover {
        background-color: #f0fdf4;
        border-color: #86efac;
        transform: translateY(-1px);
    }
    .icon-circle {
        flex-shrink: 0; width: 2.75rem; height: 2.75rem;
        background: white; border-radius: 0.75rem;
        display: flex; align-items: center; justify-content: center;
        color: #16a34a !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        border: 1px solid #dcfce7;
    }

    .progress-shell {
        background-color: #e5e7eb; border-radius: 9999px;
        height: 0.85rem; overflow: hidden; margin-top: 0.5rem;
    }
    .progress-core {
        height: 100%;
        background: linear-gradient(to right, #22c55e, #059669);
        border-radius: 9999px;
    }

    /* Footer Disease Grid */
    .d-grid {
        display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 1.25rem; margin-top: 1rem;
    }
    .d-box {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(6px); border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.25rem; border-radius: 1rem; color: white !important;
    }
    .d-box h4, .d-box p { color: white !important; }

    .step-ball {
        width: 2.75rem; height: 2.75rem;
        background-color: #dcfce7; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 0.75rem; color: #15803d !important;
        font-weight: 800; font-size: 1.15rem;
        border: 1px solid #bbf7d0;
    }
    
    /* Sidebar Widgets override */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {
        color: #064e3b !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #064e3b !important;
    }
</style>
""", unsafe_allow_html=True)

# --- ICONS ---
SVG_LEAF = """<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: white;"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 3.5-.4 9.2-1.1 2.7-3.3 4.7-5.8 6.3-.3.2-.6.3-.8.5Z"/><path d="M11 20c1.1-2.8 2.4-5.5 4-8"/><path d="M11 20a8 8 0 0 0 3.5-15.5"/></svg>"""
SVG_CHECK = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>"""
SVG_WARN = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""
SVG_CRIT = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>"""

# SVG Resources for Recommendations
REC_SVGS = {
    'drop': """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 16.3c2.2 0 4-1.8 4-4 0-3.3-4-6.3-4-6.3S3 9 3 12.3c0 2.2 1.8 4 4 4z"/><path d="M17 20.8c2.2 0 4-1.8 4-4 0-3.3-4-6.3-4-6.3S13 13.5 13 16.8c0 2.2 1.8 4 4 4z"/></svg>""",
    'sun': """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="M4.93 4.93l1.41 1.41"/><path d="M17.66 17.66l1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="M6.34 17.66l-1.41 1.41"/><path d="M19.07 4.93l-1.41 1.41"/></svg>""",
    'shield': """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 .76-.97l8-2a1 1 0 0 1 .48 0l8 2A1 1 0 0 1 20 6v7Z"/></svg>""",
    'spark': """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.937A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .962 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.582a.5.5 0 0 1 0 .962L15.5 14.063A2 2 0 0 0 14.063 15.5l-1.582 6.135a.5.5 0 0 1-.962 0L9.937 15.5Z"/><path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/></svg>""",
    'scissor': """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="6" cy="6" r="3"/><path d="M8.12 8.12 12 12"/><circle cx="6" cy="18" r="3"/><path d="M8.12 15.88 12 12"/><path d="M16 12 8.12 19.88"/><path d="M22 6 12 16"/></svg>""",
    'wind': """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/></svg>"""
}

# --- DATA & CONFIG ---
DIAGNOSIS_LABELS = [
    'Antraknosa',
    'Bacterial Canker',
    'Sehat',
    'Embun Tepung'
]

DETAILED_INFO = {
    'Antraknosa': {
        'severity': 'severe',
        'description': 'Antraknosa dipicu oleh jamur Colletotrichum gloeosporioides. Ciri khasnya adalah bercak hitam nekrotik yang meluas di daun dan membentuk lesi cekung pada buah. Penyakit ini sangat menular melalui percikan air hujan dan kelembapan tinggi, memicu kerontokan daun masif jika tidak ditangani.'
    },
    'Bacterial Cancer': {
        'severity': 'severe',
        'description': 'Dikenal juga sebagai Kanker Bakteri Mangga (Xanthomonas axonopodis). Gejala berupa bercak basah (water-soaked) berbentuk bersudut (angular) yang lambat laun mengering, menebal, dan menyerupai kerak pecah-pecah pada daun dan batang. Bakteri ini bersifat sistemik dan menyerang jaringan vaskular pohon.'
    },
    'Embun Tepung': {
        'severity': 'moderate',
        'description': 'Infeksi jamur Oidium mangiferae yang menyelimuti permukaan atas daun, malai bunga, dan buah muda dengan lapisan miselium putih menyerupai bedak. Menghambat fotosintesis total, memicu kerontokan bunga (gagal berbuah), dan paling aktif pada kondisi malam lembap dan siang terik.'
    },
    'Sehat': {
        'severity': 'mild',
        'description': 'Daun dalam kondisi metabolisme prima. Klorofil merata, tidak ada lesi nekrotik, jamur superficial, maupun bakteri vaskular. Pertumbuhan jaringan baru berjalan normal.'
    }
}

def get_treatments(disease, severity):
    if "Sehat" in disease:
        return [
            {'icon': REC_SVGS['drop'], 'title': 'Manajemen Air & Irigasi', 'desc': 'Sesuaikan penyiraman dengan fase pertumbuhan. Kurangi saat pembungaan agar bunga tidak rontok, jaga drainase agar air tidak menggenang.'},
            {'icon': REC_SVGS['sun'], 'title': 'Pemangkasan Pemeliharaan', 'desc': 'Buang tunas air yang tumbuh di batang utama dan cabang yang tumpang tindih untuk memastikan cahaya merata di seluruh kanopi.'},
            {'icon': REC_SVGS['spark'], 'title': 'Pemupukan Terjadwal (NPK)', 'desc': 'Berikan pupuk NPK 15:15:15 dicampur pupuk organik matang/kompos di awal dan akhir musim hujan di sekitar lingkaran tajuk pohon.'},
            {'icon': REC_SVGS['shield'], 'title': 'Monitoring Rutin', 'desc': 'Lakukan inspeksi fisik minimal seminggu sekali pada permukaan bawah daun untuk mendeteksi dini telur hama atau gejala penyakit.'}
        ]
    
    treat = []
    if "Antraknosa" in disease:
        treat = [
            {'icon': REC_SVGS['scissor'], 'title': 'Sanitasi Keras & Pembakaran', 'desc': 'Potong ranting terinfeksi minimal 15-20cm di bawah batas gejala. Bakar sisa potongan jauh dari kebun untuk memutus siklus spora.'},
            {'icon': REC_SVGS['shield'], 'title': 'Fungisida Protektif Spesifik', 'desc': 'Semprot rutin dengan bahan aktif Mankozeb atau Tembaga Oksiklorida setiap 7-10 hari, terutama menjelang musim hujan.'},
            {'icon': REC_SVGS['wind'], 'title': 'Modifikasi Iklim Mikro', 'desc': 'Lakukan pemangkasan kanopi agar sinar matahari masuk dan sirkulasi udara lancar guna menurunkan kelembapan di area pohon.'},
            {'icon': REC_SVGS['spark'], 'title': 'Nutrisi Daya Tahan', 'desc': 'Berikan pupuk tinggi Kalium (K) untuk mempertebal dinding sel daun sehingga lebih sulit ditembus miselium jamur.'},
            {'icon': REC_SVGS['drop'], 'title': 'Pascapanen Hot-Water', 'desc': 'Untuk buah, lakukan perendaman air panas (52°C) selama 5 menit untuk membunuh spora laten yang terbawa.'}
        ]
    elif "Bacterial Cancer" in disease:
        treat = [
            {'icon': REC_SVGS['shield'], 'title': 'Bakterisida Tembaga Hidroksida', 'desc': 'Semprotkan bakterisida berbahan aktif Tembaga Hidroksida (dosis 2g/Liter) atau antibiotik tanaman Streptomycin secara berkala.'},
            {'icon': REC_SVGS['scissor'], 'title': 'Bedah Luka Batang & Plester', 'desc': 'Kupas bagian kulit kayu yang terkena kanker dengan pisau bersih, lalu oleskan pasta fungisida pekat atau Bubur Bordo (Bordeaux Mixture).'},
            {'icon': REC_SVGS['spark'], 'title': 'Sterilisasi Alat 100%', 'desc': 'Celupkan alat potong ke dalam Alkohol 70% atau air sabun setiap kali pindah memotong dari satu dahan ke dahan lain agar bakteri tidak tertular.'},
            {'icon': REC_SVGS['drop'], 'title': 'Manajemen Hama Vektor', 'desc': 'Kendalikan serangga penggigit seperti wereng mangga yang luka bekas gigitannya sering menjadi pintu masuk bakteri Xanthomonas.'}
        ]
    elif "Embun Tepung" in disease:
        treat = [
            {'icon': REC_SVGS['shield'], 'title': 'Aplikasi Sulfur Wettable', 'desc': 'Gunakan belerang/sulfur cair (2-4g/L). Semprot pada pagi hari saat cuaca cerah, hindari siang terik untuk mencegah phytotoxic (daun terbakar).'},
            {'icon': REC_SVGS['spark'], 'title': 'Fungisida Sistemik Triazol', 'desc': 'Jika serangan di malai bunga parah, gunakan bahan aktif Difenokonazol atau Triadimefon sesuai dosis label saat 50% bunga mulai mekar.'},
            {'icon': REC_SVGS['sun'], 'title': 'Penetrasi Radiasi Matahari', 'desc': 'Jamur Oidium sangat sensitif terhadap UV. Buka rongga udara tengah kanopi pohon untuk memaksimalkan sinar matahari menembus area dalam.'},
            {'icon': REC_SVGS['drop'], 'title': 'Kontrol Nutrisi Nitrogen', 'desc': 'Hentikan sementara pemberian pupuk tinggi Urea/Nitrogen berlebih karena jaringan daun yang terlalu lunak sangat disukai embun tepung.'}
        ]
    
    return treat

def get_sev_props(sev):
    if sev == 'severe':
        return {'bg': '#fef2f2', 'brd': '#fecaca', 'txt': '#b91c1c', 'ico': '#dc2626', 'badge': '#fee2e2', 'lbl': 'Parah', 'svg': SVG_CRIT}
    elif sev == 'moderate':
        return {'bg': '#fffbeb', 'brd': '#fde68a', 'txt': '#b45309', 'ico': '#d97706', 'badge': '#fef3c7', 'lbl': 'Sedang', 'svg': SVG_WARN}
    return {'bg': '#f0fdf4', 'brd': '#bbf7d0', 'txt': '#15803d', 'ico': '#16a34a', 'badge': '#dcfce7', 'lbl': 'Ringan', 'svg': SVG_CHECK}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Status Model AI")
    
    MODEL_NAME = "model fix.pth"
    arch_type = "ResNet50"
    
    
    active_mode = "Simulasi"
    loaded_model = None
    
    if os.path.exists(MODEL_NAME):
        with st.spinner("🧠 Memuat Arsitektur ResNet50..."):
            mod = load_prediction_model(MODEL_NAME)
            if isinstance(mod, Exception):
                st.error(f"❌ Gagal memuat file model: {str(mod)}")
            else:
                loaded_model = mod
                active_mode = "Real"
                st.success("✅ Sistem AI ResNet50 Aktif!")
                st.info(f"Memuat: `{MODEL_NAME}`")
    else:
        st.warning(f"⚠️ Model `{MODEL_NAME}` belum ada di server.")
        
        if not os.path.exists(MODEL_NAME):

    with st.status(
        "📥 Downloading model...",
        expanded=True
    ):

        try:

            response = requests.get(
                MODEL_URL,
                stream=True,
                timeout=300
            )

            response.raise_for_status()

            with open(MODEL_NAME, "wb") as f:

                for chunk in response.iter_content(
                    chunk_size=8192
                ):
                    if chunk:
                        f.write(chunk)

            st.success(
                "✅ Model berhasil didownload"
            )

        except Exception as e:

            st.error(
                f"❌ Gagal download model: {e}"
            )

    st.markdown("---")
    st.caption("**Order Label Output (Model):**")
    for idx, label in enumerate(DIAGNOSIS_LABELS):
        st.caption(f"**[{idx}]** {label}")

# --- HEADER BANNER ---
st.markdown(f"""
<div class="fancy-header">
    <div class="header-icon-box">{SVG_LEAF}</div>
    <div>
        <div class="header-title-text">Deteksi Penyakit Daun Mangga</div>
        <div class="header-subtitle-text">Sistem diagnosa cerdas kesehatan tanaman berbasis Deep Learning</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- LAYOUT ---
if 'uploaded_image' not in st.session_state: st.session_state.uploaded_image = None
if 'diagnosis' not in st.session_state: st.session_state.diagnosis = None

colL, colR = st.columns([1, 1], gap="medium")

# --- LEFT: INPUT ---
with colL:
    with st.container(border=True):
        st.markdown("#### 📸 Unggah Daun Mangga")
        st.markdown("Foto jelas diperlukan untuk akurasi deteksi.")
        
        tab1, tab2 = st.tabs(["📁 Unggah File", "📷 Ambil Foto"])
        
        with tab1:
            u_file = st.file_uploader("Pilih Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="file_source")
        with tab2:
            c_file = st.camera_input("Ambil Foto", label_visibility="collapsed", key="cam_source")
            
        final_source = c_file if c_file is not None else u_file
        
        if final_source:
            img = Image.open(final_source)
            
            # Generate unique signature to detect change
            src_id = getattr(final_source, "name", "camera")
            src_size = getattr(final_source, "size", random.randint(0,999999))
            mid = f"{src_id}_{src_size}_fixed"
            
            if 'curr_id' not in st.session_state or st.session_state.curr_id != mid:
                st.session_state.curr_id = mid
                st.session_state.uploaded_image = img
                
                if active_mode == "Real" and loaded_model:
                    with st.spinner("AI Menganalisis..."):
                        try:
                            tensor = preprocess_image(img)
                            with torch.no_grad():
                                output = loaded_model(tensor)
                                probs = torch.softmax(output, dim=1)
                                confidence, pred = torch.max(probs, 1)
                            idx = pred.item()
                            cf = confidence.item() * 100
                            d_name = DIAGNOSIS_LABELS[idx]
                            st.session_state.diagnosis = {
                                'd': d_name, 'cf': round(cf, 1), 
                                'src': f"AI ({arch_type})", 'meta': DETAILED_INFO[d_name]
                            }
                        except Exception as er:
                            st.error(f"Prediksi Gagal: {str(er)}")
                else:
                    with st.spinner("Simulasi Analisis..."):
                        time.sleep(1.5)
                        d_name = random.choice(DIAGNOSIS_LABELS)
                        st.session_state.diagnosis = {
                            'd': d_name, 'cf': round(85 + random.uniform(0, 14), 1), 
                            'src': "Simulasi", 'meta': DETAILED_INFO[d_name]
                        }
        else:
            st.session_state.diagnosis = None
            st.session_state.uploaded_image = None
            if 'curr_id' in st.session_state: del st.session_state.curr_id

    if st.session_state.uploaded_image:
        with st.container(border=True):
            st.markdown("##### 🖼️ Preview Gambar")
            st.image(st.session_state.uploaded_image, use_container_width=True)

# --- RIGHT: RESULT ---
with colR:
    if st.session_state.diagnosis:
        dat = st.session_state.diagnosis
        m = dat['meta']
        p = get_sev_props(m['severity'])
        
        # Result Card using Streamlit container for unified layout
        with st.container(border=True):
            # Custom inner HTML
            st.markdown(f"""
<div style="background-color: {p['bg']}; border-bottom: 1px solid {p['brd']}; padding: 1rem; border-radius: 0.75rem; display: flex; justify-content: space-between; align-items: center;">
<div style="display: flex; align-items: center; gap: 0.75rem;">
<div style="color: {p['ico']};">{p['svg']}</div>
<div>
<div style="font-weight: 700; color: #064e3b;">Hasil Diagnosa</div>
<div style="font-size: 0.8rem; color: #15803d;">Sumber: {dat['src']}</div>
</div>
</div>
<span style="background: {p['badge']}; padding: 0.2rem 0.7rem; border-radius: 2rem; font-size: 0.75rem; font-weight:600;">{p['lbl']}</span>
</div>
<div style="margin-top: 1.25rem;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
<span style="font-weight: 600; font-size: 0.9rem;">Penyakit Terdeteksi</span>
<span style="font-size: 0.85rem; color: #15803d;">{dat['cf']}% confidence</span>
</div>
<div class="fancy-result-banner">{dat['d']}</div>
<div style="margin-top: 1.25rem;">
<div class="progress-shell">
<div class="progress-core" style="width: {dat['cf']}%"></div>
</div>
</div>
<div style="margin-top: 1.25rem;">
<div style="font-weight:600; margin-bottom: 0.25rem; font-size:0.9rem;">Keterangan</div>
<div style="font-size: 0.9rem; color: #065f46; line-height:1.5;">{m['description']}</div>
</div>
</div>
""", unsafe_allow_html=True)
            
        # Treatments Card
        with st.container(border=True):
            st.markdown(f"""
<div style="background: linear-gradient(to right, #16a34a, #059669); padding: 1rem; border-radius: 0.75rem; color: white !important; margin-bottom: 1rem;">
    <div style="font-weight:700; color:white;">🌿 Rekomendasi Penanganan</div>
    <div style="font-size: 0.8rem; color: #dcfce7;">Langkah spesifik mitigasi penyakit</div>
</div>
""", unsafe_allow_html=True)
            
            treats = get_treatments(dat['d'], m['severity'])
            for t in treats:
                st.markdown(f"""
<div class="item-card-inner">
    <div class="icon-circle">{t['icon']}</div>
    <div>
        <div style="font-weight: 700; font-size: 0.95rem;">{t['title']}</div>
        <div style="font-size: 0.85rem; color: #065f46; line-height: 1.4;">{t['desc']}</div>
    </div>
</div>
""", unsafe_allow_html=True)
    else:
        with st.container(border=True):
            st.markdown("""
<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 300px; opacity: 0.5;">
    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 3.5-.4 9.2-1.1 2.7-3.3 4.7-5.8 6.3-.3.2-.6.3-.8.5Z"/><path d="M11 20c1.1-2.8 2.4-5.5 4-8"/><path d="M11 20a8 8 0 0 0 3.5-15.5"/></svg>
    <div style="margin-top: 1rem; font-weight: 500;">Silakan unggah gambar untuk memulai</div>
</div>
""", unsafe_allow_html=True)

# --- FOOTER INFO ---
with st.container(border=True):
    st.markdown("#### 💡 Cara Kerja")
    i_col1, i_col2, i_col3 = st.columns(3)
    with i_col1:
        st.markdown('<div align="center"><div class="step-ball">1</div><b>Unggah Gambar</b><div style="font-size:0.85rem; margin-top:0.2rem;">Input foto daun terpapar</div></div>', unsafe_allow_html=True)
    with i_col2:
        st.markdown('<div align="center"><div class="step-ball">2</div><b>Eksekusi AI</b><div style="font-size:0.85rem; margin-top:0.2rem;">Model CNN memproses gambar</div></div>', unsafe_allow_html=True)
    with i_col3:
        st.markdown('<div align="center"><div class="step-ball">3</div><b>Hasil & Solusi</b><div style="font-size:0.85rem; margin-top:0.2rem;">Diagnosa penyakit terbit</div></div>', unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #16a34a, #047857); border-radius: 1rem; padding: 2rem; margin-top: 1.5rem;">
    <h3 style="color: white !important; margin-top: 0;">📌 Referensi Penyakit Umum</h3>
    <div class="d-grid">
        <div class="d-box"><h4>Antraknosa</h4><p>Bercak hitam pada daun.</p></div>
        <div class="d-box"><h4>Bacterial Cancer</h4><p>Bercak basah dan berkerak.</p></div>
        <div class="d-box"><h4>Embun Tepung</h4><p>Lapisan putih seperti bedak.</p></div>
        <div class="d-box"><h4>Sehat</h4><p>Daun hijau segar normal.</p></div>
    </div>
</div>
""", unsafe_allow_html=True)
