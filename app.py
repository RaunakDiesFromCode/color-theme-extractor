import json

import numpy as np
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from theme_extractor import ThemeExtractor


def render_copy_button(label, text_payload):
    js_payload = json.dumps(text_payload)
    components.html(
        f"""
        <button onclick='navigator.clipboard.writeText({js_payload})'
        style="
            background:#1f7a5c;
            color:white;
            padding:10px 14px;
            border:none;
            border-radius:8px;
            cursor:pointer;
            font-weight:600;
            width:100%;
        ">
        {label}
        </button>
    """,
        height=52,
    )


def make_palette_strip(palette_rgb, width=720, height=70):
    palette = np.zeros((height, width, 3), dtype=float)
    if not palette_rgb:
        return palette

    step = max(1, width // len(palette_rgb))
    for i, rgb in enumerate(palette_rgb):
        start = i * step
        end = width if i == len(palette_rgb) - 1 else (i + 1) * step
        palette[:, start:end, :] = np.array(rgb) / 255.0
    return palette


st.set_page_config(layout="wide", page_title="Theme Palette Engine")

st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
.card {
    padding: 18px;
    border-radius: 14px;
    background: #101521;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 16px;
}
.card-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 8px;
}
.card-sub {
    color: #b5becc;
    font-size: 14px;
    margin-bottom: 14px;
}
.footer {
    margin-top: 16px;
    text-align: center;
    font-size: 13px;
    color: #8b96aa;
    padding: 12px 8px;
    border-top: 1px solid rgba(255,255,255,0.08);
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Image to Theme Palette")
st.caption(
    "Upload an image to generate a weighted, usability-aware palette and UI theme."
)

st.sidebar.header("Engine Controls")
k = st.sidebar.slider(
    "Clusters (K)",
    min_value=3,
    max_value=10,
    value=6,
    help="How many dominant color groups to extract.",
)
population_weight = st.sidebar.slider(
    "Population Weight",
    min_value=0.0,
    max_value=1.5,
    value=0.5,
    help="Higher values favor colors that occupy larger image regions.",
)
saturation_weight = st.sidebar.slider(
    "Saturation Weight",
    min_value=0.0,
    max_value=2.0,
    value=0.8,
    help="Higher values favor more vivid colors.",
)
penalty_enabled = st.sidebar.checkbox(
    "Penalize too dark/light colors",
    value=True,
    help="Applies a penalty to colors with poor lightness usability.",
)

uploaded = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
)

if uploaded:
    extractor = ThemeExtractor(
        k=k,
        population_weight=population_weight,
        saturation_weight=saturation_weight,
        penalty_enabled=penalty_enabled,
    )
    result = extractor.extract(uploaded)

    original = result["images"]["original"]
    resized = result["images"]["resized"]
    clustered = result["images"]["clustered"]
    palette_rgb = result["palette_rgb"]
    accent_rgb = result["accent_rgb"]
    cluster_details = result["cluster_details"]
    theme = result["theme"]

    theme_json = json.dumps(theme, indent=2)
    palette_strip = make_palette_strip(palette_rgb)
    accent_patch = np.ones((120, 120, 3), dtype=float) * \
        (np.array(accent_rgb) / 255.0)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Image Input and Theme Palette</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="card-sub">Top priority output: image preview, extracted palette, accent choice, and theme JSON actions.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.image(original, caption="Original Image", use_container_width=True)
    c2.image(
        resized, caption=f"Resized for Analysis ({result['config']['resize'][0]}x{result['config']['resize'][1]})", use_container_width=True)

    c3, c4 = st.columns([3, 1])
    c3.image(palette_strip, caption="Extracted Palette (Dominance Sorted)",
             use_container_width=True)
    c4.image(accent_patch,
             caption=f"Accent {accent_rgb}", use_container_width=True)

    st.write("Theme JSON payload:")
    with st.popover("Preview Theme JSON"):
        st.code(theme_json, language="json")

    c5, c6 = st.columns(2)
    with c5:
        render_copy_button("Copy Theme JSON", theme_json)
    with c6:
        st.download_button(
            "Download Theme JSON",
            data=theme_json,
            file_name="theme.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">How It Works (Detailed)</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
1. **Preprocessing**: The image is resized to reduce computation while preserving color distribution.
2. **Perceptual conversion**: RGB pixels are converted to LAB and HSV.
   - LAB is used for clustering because Euclidean distance in LAB better matches human color similarity.
   - HSV saturation is used as a visual strength signal.
3. **K-Means clustering**: Pixels in LAB are grouped into $K$ dominant clusters.
4. **Dominance sorting**: Clusters are sorted by pixel count, so the palette starts with the most dominant color.
5. **Feature extraction per cluster**:
   - Population $p_i = \frac{count_i}{\sum count}$
   - Saturation $s_i = \text{mean HSV saturation of cluster } i$
   - Lightness $L_i = \text{LAB L channel of cluster center}$
6. **Usability-aware scoring**:
   - Lightness penalty is applied if color is too dark/light for broad UI use.
   - Score formula:

$$
\text{score}_i = \left(w_{pop} \cdot p_i + w_{sat} \cdot s_i\right) \cdot penalty_i
$$

   - If penalty is enabled and $L_i < 20$ or $L_i > 85$, then $penalty_i = 0.5$, else $1.0$.
7. **Accent selection**: The highest scoring cluster center becomes the accent color.
8. **Theme synthesis**: Accent hue anchors derived colors (`primary`, `secondary`, `background_light`, `background_dark`, `surface`) plus contrast-safe text color.
"""
    )

    lab_ab = result["distribution"]["lab_ab"]
    rgb_norm = result["distribution"]["rgb_norm"]
    color_values = [
        f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
        for r, g, b in rgb_norm
    ]
    fig = px.scatter(
        x=lab_ab[:, 0],
        y=lab_ab[:, 1],
        color=color_values,
        title="LAB Distribution (A vs B)",
    )
    st.plotly_chart(fig, use_container_width=True)

    c7, c8 = st.columns(2)
    c7.image(clustered, caption="Clustered Preview", use_container_width=True)

    theme_preview = np.zeros((120, 600, 3), dtype=float)
    swatches = [
        np.array(theme["primary"]) / 255.0,
        np.array(theme["secondary"]) / 255.0,
        np.array(theme["background_light"]) / 255.0,
        np.array(theme["background_dark"]) / 255.0,
        np.array(theme["surface"]) / 255.0,
    ]
    step = 600 // len(swatches)
    for i, swatch in enumerate(swatches):
        start = i * step
        end = 600 if i == len(swatches) - 1 else (i + 1) * step
        theme_preview[:, start:end, :] = swatch
    c8.image(theme_preview, caption="Generated Theme",
             use_container_width=True)

    table = []
    for cluster in cluster_details:
        table.append(
            {
                "Rank": cluster["rank"],
                "RGB": str(cluster["rgb"]),
                "Count": cluster["count"],
                "Population": round(cluster["population"], 4),
                "Saturation": round(cluster["saturation"], 4),
                "Lightness": round(cluster["lightness"], 2),
                "Score": round(cluster["score"], 4),
            }
        )
    st.dataframe(table, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Upload an image to begin extraction.")

st.markdown('<div class="footer">Made by Raunak | GitHub: RaunakDiesFromCode</div>',
            unsafe_allow_html=True)
