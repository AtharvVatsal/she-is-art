import streamlit as st
from PIL import Image
from pathlib import Path

from utils import stylize
from captions import generate_caption

# ——— Streamlit page config ———
st.set_page_config(
    page_title="She is Art 🎨",
    page_icon="🖌️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ——— Header ———
st.markdown(
    "<h1 style='text-align:center;color:#e75480;font-family:Georgia;'>She is Art</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;font-size:1.1rem;'>Upload her photo and see the world the way your heart paints her.</p>",
    unsafe_allow_html=True
)

# ——— Discover style models dynamically ———
styles_dir = Path(__file__).parent / "styles"
style_models = {
    p.stem.replace("_", " ").title(): str(p)
    for p in styles_dir.glob("*.pth")
}

if not style_models:
    st.error(
        "No style models found in the `styles/` directory. "
        "Please add your `.pth` files there."
    )
    st.stop()

# ——— User inputs ———
style_choice = st.selectbox("Select an art style:", list(style_models.keys()))
uploaded_file = st.file_uploader("Choose an image…", type=["jpg", "jpeg", "png"])

# ——— Main functionality ———
if uploaded_file:
    # Load and display original
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Original", use_column_width=True)

    # Generate stylized output
    if st.button("Generate Art"):
        with st.spinner("Painting with affection…"):
            model_path = style_models[style_choice]
            out_img = stylize(img, model_path)
            st.image(out_img, caption=f"Stylized ({style_choice})", use_column_width=True)

            # Generate and display a romantic caption
            caption = generate_caption()
            st.markdown(
                f"<p style='text-align:center;font-style:italic;color:#cc3366;'>{caption}</p>",
                unsafe_allow_html=True
            )
