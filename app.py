import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import requests
from streamlit_lottie import st_lottie
from deep_translator import GoogleTranslator
from gtts import gTTS
import io

st.set_page_config(
    page_title="KX Image Capioning ", 
    page_icon="KC", 
    layout="wide"
)
st.markdown("""
<style>
    /* Import a nice font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    html, body, [class*="st-"], [class*="css-"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Center the main title */
    [data-testid="stAppViewContainer"] > .main > div:first-child > div:first-child > div:first-child {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    /* Style the two main columns as "cards" */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: rgba(255, 255, 255, 0.05); /* Semi-transparent white */
        border-radius: 15px; /* Rounded corners */
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5); /* 3D-like shadow */
        border: 1px solid rgba(255, 255, 255, 0.1); /* Subtle border */
    }

    /* Center headers inside the columns */
    h1, h2, h3 {
        text-align: center;
    }
    
    /* Make the title bigger */
    h1 {
        font-size: 2.5rem !important;
    }
    
    /* --- [NEW] Color Styling --- */
    
    /* Make the main caption result blue */
    [data-testid="stMarkdown"] > div > p > strong {
        font-size: 1.25rem;
        color: #00A9FF; /* A nice, bright blue */
        font-weight: 600;
    }
    
    /* Make the "In [Language]:" text blue */
    [data-testid="stMarkdown"] > div > p > strong:not(:first-child) {
         color: #00A9FF;
    }
    
    /* Make the translated text white */
    [data-testid="stInfo"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        color: white; /* Force white text */
    }
    
    /* Hide the "Deploy" button */
    [data-testid="stDeployButton"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation_url = "https://lottie.host/8a729107-1c1a-4318-8316-168a2e1d7570/310kcsi4l5.json"
lottie_anim = load_lottieurl(lottie_animation_url)

@st.cache_resource
def load_models():
    print("Loading all models...")
   
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    object_detector = pipeline('object-detection', model='facebook/detr-resnet-50')
    
    sentiment_analyzer = pipeline('sentiment-analysis')
    
    print("All models loaded!")
    return processor, model, device, object_detector, sentiment_analyzer

processor, model, device, object_detector, sentiment_analyzer = load_models()

with st.container():
    st.title("Karmatix Image Captioning")
    if lottie_anim:
        st_lottie(lottie_anim, speed=1, height=200, key="initial_animation")

st.write("---") 

col1, col2 = st.columns(2, gap="large") 

with col1:
    st.header("1. Upload Your Image")
    uploaded_file = st.file_uploader("Upload an image to get started...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

with col2:
    st.header("2. Get Your Result")
    
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.image(raw_image, caption='Your Uploaded Image', use_container_width=True)
        
        st.write("Generating caption...")
        
        inputs = processor(raw_image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        predicted_caption = processor.decode(out[0], skip_special_tokens=True)
        
        st.subheader("Generated Caption:")
        st.markdown(f"**{predicted_caption}**") 
        
        sentiment = sentiment_analyzer(predicted_caption)[0]
        st.write(f"**Caption Vibe:** {sentiment['label']} (Score: {sentiment['score']:.2f})")
        
        st.divider()

        st.subheader("Objects Detected:")
        objects = object_detector(raw_image)
        if objects:
            detected_labels = [obj['label'] for obj in objects]
            unique_tags = {f"`{label}` ({detected_labels.count(label)})" for label in set(detected_labels)}
            st.write("I can see: " + ", ".join(unique_tags))
        else:
            st.write("No specific objects detected.")
        
        st.divider()
        
        st.subheader("3. Translate & Listen")
        
        lang_options = {
            'English': 'en',
            'Hindi (हिंदी)': 'hi',
            'Spanish (Español)': 'es',
            'French (Français)': 'fr',
            'German (Deutsch)': 'de'
        }
        lang_choice_key = st.selectbox("Choose a language:", options=lang_options.keys())
        lang_code = lang_options[lang_choice_key]

        try:
            translated_text = GoogleTranslator(source='auto', target=lang_code).translate(predicted_caption)
        except Exception as e:
            st.error(f"Translation failed: {e}")
            translated_text = predicted_caption
            
        st.write(f"**In {lang_choice_key}:**")
        st.info(translated_text) 

        try:
            tts = gTTS(text=translated_text, lang=lang_code)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            st.audio(audio_bytes, format='audio/mp3')
        except Exception as e:
            st.error(f"Audio generation failed: {e}")
            
    else:
        st.info("Upload an image on the left to get started.")