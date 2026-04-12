import io
import json
import os
from typing import Any, Dict, List

import anthropic
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

st.set_page_config(page_title="AI Image Editor", page_icon="🖼️", layout="wide")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def local_instruction_parser(user_prompt: str) -> Dict[str, Any]:
    text = user_prompt.lower()
    operations: List[Dict[str, Any]] = []

    if "grayscale" in text or "black and white" in text or "b&w" in text:
        operations.append({"type": "grayscale"})
    if "blur" in text:
        operations.append({"type": "blur", "radius": 1.8})
    if "sharpen" in text or "sharp" in text:
        operations.append({"type": "sharpness", "value": 1.4})
    if "bright" in text or "brightness" in text:
        operations.append({"type": "brightness", "value": 1.2})
    if "dark" in text:
        operations.append({"type": "brightness", "value": 0.85})
    if "contrast" in text:
        operations.append({"type": "contrast", "value": 1.25})
    if "saturat" in text or "vibrant" in text:
        operations.append({"type": "saturation", "value": 1.25})
    if "flip horizontal" in text or "mirror" in text:
        operations.append({"type": "flip_horizontal"})
    if "flip vertical" in text:
        operations.append({"type": "flip_vertical"})
    if "rotate left" in text:
        operations.append({"type": "rotate", "degrees": 90})
    if "rotate right" in text:
        operations.append({"type": "rotate", "degrees": -90})

    return {"operations": operations}


def ai_instruction_parser(user_prompt: str, api_key: str) -> Dict[str, Any]:
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""
Convert this user image editing request into strict JSON with a top-level key "operations".
Allowed operation types only:
- brightness (value: 0.4 to 2.0)
- contrast (value: 0.4 to 2.0)
- saturation (value: 0.0 to 2.0)
- sharpness (value: 0.0 to 3.0)
- blur (radius: 0.0 to 8.0)
- grayscale
- rotate (degrees: -180 to 180)
- flip_horizontal
- flip_vertical

Return JSON only. No extra text.

User request: {user_prompt}
"""
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    content = message.content[0].text.strip()
    return json.loads(content)


def apply_operations(image: Image.Image, operations: List[Dict[str, Any]]) -> Image.Image:
    edited = image.convert("RGB")
    for op in operations:
        op_type = op.get("type")
        if op_type == "brightness":
            factor = _clamp(float(op.get("value", 1.0)), 0.4, 2.0)
            edited = ImageEnhance.Brightness(edited).enhance(factor)
        elif op_type == "contrast":
            factor = _clamp(float(op.get("value", 1.0)), 0.4, 2.0)
            edited = ImageEnhance.Contrast(edited).enhance(factor)
        elif op_type == "saturation":
            factor = _clamp(float(op.get("value", 1.0)), 0.0, 2.0)
            edited = ImageEnhance.Color(edited).enhance(factor)
        elif op_type == "sharpness":
            factor = _clamp(float(op.get("value", 1.0)), 0.0, 3.0)
            edited = ImageEnhance.Sharpness(edited).enhance(factor)
        elif op_type == "blur":
            radius = _clamp(float(op.get("radius", 1.0)), 0.0, 8.0)
            edited = edited.filter(ImageFilter.GaussianBlur(radius=radius))
        elif op_type == "grayscale":
            edited = ImageOps.grayscale(edited).convert("RGB")
        elif op_type == "rotate":
            degrees = _clamp(float(op.get("degrees", 0.0)), -180.0, 180.0)
            edited = edited.rotate(degrees, expand=True)
        elif op_type == "flip_horizontal":
            edited = ImageOps.mirror(edited)
        elif op_type == "flip_vertical":
            edited = ImageOps.flip(edited)
    return edited


st.title("🖼️ AI Image Editor")
st.caption("Upload an image, describe edits in plain language, and generate an edited version.")

with st.sidebar:
    st.subheader("Settings")
    api_key_input = st.text_input("Anthropic API Key (optional)", type="password")
    use_ai = st.toggle("Use AI parser", value=True)
    st.info("If API key is not set, the app uses a safe local parser with common edit commands.")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
with col2:
    user_prompt = st.text_area(
        "Describe required changes",
        placeholder="Example: Make the image brighter, increase contrast a bit, and blur the background lightly.",
        height=140,
    )

if uploaded_file:
    original_img = Image.open(uploaded_file)
    st.markdown("### Original Image")
    st.image(original_img, use_container_width=True)

if uploaded_file and user_prompt:
    if st.button("✨ Apply AI Edits"):
        with st.spinner("Understanding instruction and editing image..."):
            try:
                api_key = api_key_input or os.getenv("ANTHROPIC_API_KEY", "")
                if use_ai and api_key:
                    parsed = ai_instruction_parser(user_prompt, api_key)
                else:
                    parsed = local_instruction_parser(user_prompt)
            except Exception:
                parsed = local_instruction_parser(user_prompt)

            operations = parsed.get("operations", [])
            if not operations:
                st.warning("No supported edits were detected from the prompt.")
            else:
                edited_img = apply_operations(original_img, operations)
                st.markdown("### Edited Image")
                st.image(edited_img, use_container_width=True)
                st.markdown("### Applied Operations")
                st.json(operations)

                output = io.BytesIO()
                edited_img.save(output, format="PNG")
                output.seek(0)
                st.download_button(
                    label="⬇️ Download Edited Image",
                    data=output,
                    file_name="edited_image.png",
                    mime="image/png",
                )
else:
    st.info("Upload an image and describe edits to start.")
