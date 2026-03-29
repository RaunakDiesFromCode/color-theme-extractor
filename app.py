import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, rgb2hsv, lab2rgb, hsv2rgb
import plotly.express as px
import json
import streamlit.components.v1 as components


def copy_button(label, data):
    json_data = json.dumps(data)

    components.html(f"""
        <button onclick='navigator.clipboard.writeText(`{json_data}`)'
        style="
            background:#2563eb;
            color:white;
            padding:8px 12px;
            border:none;
            border-radius:8px;
            cursor:pointer;
            margin-top:8px;
        ">
        📋 {label}
        </button>
    """, height=50)


# =========================
# PAGE CONFIG + STYLES
# =========================
st.set_page_config(layout="wide", page_title="Theme Engine")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.section {
    padding: 18px;
    border-radius: 16px;
    background: #0f1117;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 16px;
}
.title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 6px;
}
.desc {
    color: #9aa0a6;
    font-size: 14px;
    margin-bottom: 10px;
}
.badge {
    padding: 4px 10px;
    border-radius: 999px;
    background: #1f2937;
    font-size: 12px;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎨 Image → Theme Engine")
st.caption(
    "Understand how colors are extracted, evaluated, and turned into a UI theme.")

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("Controls")

K = st.sidebar.slider("Clusters (K)", 3, 10, 6,
                      help="Number of color groups extracted from the image")

w_pop = st.sidebar.slider("Population Weight", 0.0, 1.5, 0.5,
                          help="How much importance large areas get")

w_sat = st.sidebar.slider("Saturation Weight", 0.0, 2.0, 0.8,
                          help="Higher = more vibrant colors preferred")

use_penalty = st.sidebar.checkbox("Penalize too dark/light colors", True)

# =========================
# IMAGE INPUT
# =========================
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_small = img.resize((64, 64))
    img_np = np.array(img_small)

    pixels = img_np.reshape(-1, 3)
    img_norm = img_np / 255.0

    # =========================
    # SECTION 1 — INPUT
    # =========================
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="title">1. Image Preprocessing</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="desc">We shrink the image to preserve color distribution while reducing computation.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.image(img, caption="Original")
    col2.image(img_small, caption="Resized (64×64)")

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # COLOR SPACE
    # =========================
    img_lab = rgb2lab(img_norm)
    img_hsv = rgb2hsv(img_norm)

    pixels_lab = img_lab.reshape(-1, 3)
    pixels_hsv = img_hsv.reshape(-1, 3)

    # =========================
    # SECTION 2 — DISTRIBUTION
    # =========================
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="title">2. Color Distribution</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="desc">
    Colors naturally form clusters. LAB space groups visually similar colors better than RGB.
    <br><br>
    <span class="badge">Example</span> Sky pixels → tight cluster
    <span class="badge">Example</span> Skin tones → grouped region
    </div>
    """, unsafe_allow_html=True)

    fig = px.scatter(
        x=pixels_lab[:, 1],
        y=pixels_lab[:, 2],
        color=[f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
               for r, g, b in img_norm.reshape(-1, 3)],
        title="LAB Color Clusters (A vs B)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # CLUSTERING
    # =========================
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(pixels_lab)

    labels = kmeans.labels_
    centers_lab = kmeans.cluster_centers_
    centers_rgb = lab2rgb(centers_lab.reshape(1, K, 3)).reshape(K, 3)
    palette_rgb = [(centers_rgb[i] * 255).astype(int).tolist() for i in range(K)]

    st.write("Palette (RGB):", palette_rgb)

    copy_button("Copy Palette JSON", palette_rgb)

    counts = np.bincount(labels)
    idx = np.argsort(counts)[::-1]

    centers_rgb = centers_rgb[idx]
    centers_lab = centers_lab[idx]
    counts = counts[idx]

    # =========================
    # SECTION 3 — CLUSTERS
    # =========================
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="title">3. Color Clustering</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="desc">
    K-Means groups pixels into dominant colors.
    Each pixel is replaced by its cluster center.
    </div>
    """, unsafe_allow_html=True)

    clustered = centers_rgb[labels].reshape(64, 64, 3)
    st.image(clustered, caption="Clustered Image")

    # palette
    palette = np.zeros((50, 300, 3))
    step = 300 // K
    for i in range(K):
        palette[:, i*step:(i+1)*step] = centers_rgb[i]

    st.image(palette, caption="Extracted Palette")

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # SAT + LIGHT
    # =========================
    cluster_saturation = []
    for i in range(K):
        cp = pixels_hsv[labels == i]
        cluster_saturation.append(np.mean(cp[:, 1]) if len(cp) else 0)

    cluster_saturation = np.array(cluster_saturation)[idx]
    lightness = centers_lab[:, 0]

    # =========================
    # SECTION 4 — SCORING
    # =========================
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="title">4. Color Scoring</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="desc">
    Each color is evaluated based on:
    <ul>
    <li>Population (importance)</li>
    <li>Saturation (visual strength)</li>
    <li>Lightness (usability)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    scores = []
    for i in range(K):
        pop = counts[i]/np.sum(counts)
        sat = cluster_saturation[i]
        L = lightness[i]

        penalty = 0.5 if (use_penalty and (L < 20 or L > 85)) else 1.0
        score = (w_pop*pop + w_sat*sat) * penalty
        scores.append(score)

    scores = np.array(scores)
    best_idx = np.argmax(scores)

    # table
    table = []
    for i in range(K):
        rgb = (centers_rgb[i]*255).astype(int)
        table.append({
            "Color": str(rgb),
            "Count": int(counts[i]),
            "Sat": round(cluster_saturation[i], 3),
            "Light": round(lightness[i], 2),
            "Score": round(scores[i], 3)
        })

    st.dataframe(table)

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # ACCENT
    # =========================
    accent = centers_rgb[best_idx]
    accent_rgb = (accent*255).astype(int)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="title">5. Accent Selection</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="desc">The highest scoring color becomes the theme anchor.</div>',
                unsafe_allow_html=True)

    st.write("Accent Color:", accent_rgb)
    st.image(np.ones((100, 100, 3))*accent)

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # THEME
    # =========================
    accent_hsv = rgb2hsv(accent.reshape(1, 1, 3)).reshape(3)
    h, s, v = accent_hsv

    primary = accent
    secondary = hsv2rgb([h, max(0.3, s*0.6), v])
    bg_light = hsv2rgb([h, 0.1, 0.95])
    bg_dark = hsv2rgb([h, 0.2, 0.15])
    surface = hsv2rgb([h, 0.15, 0.9])

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="title">6. Theme Generation</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="desc">
    The accent hue is reused while saturation and brightness are adjusted
    to create UI-safe colors.
    </div>
    """, unsafe_allow_html=True)

    theme = np.zeros((120, 600, 3))
    colors = [primary, secondary, bg_light, bg_dark, surface]
    step = 600//len(colors)

    for i, c in enumerate(colors):
        theme[:, i*step:(i+1)*step] = c

    st.image(theme)

    st.markdown('</div>', unsafe_allow_html=True)
