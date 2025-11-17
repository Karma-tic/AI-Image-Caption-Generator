# 🚀 Karmatix AI - Image Captioning

This is a multi-featured, interactive AI web application that generates image captions and much more. This project was built using **Streamlit** (for the front-end) and multiple **Hugging Face Transformer** models (for the AI).

![Karmatix App Demo]([(https://github.com/user-attachments/assets/f89be3f4-32f6-4dd4-bf6a-344ae6267fd6))

---

## ✨ Features

* **AI Image Captioning:** Upload any image and get a detailed, descriptive caption using the **Salesforce BLIP** model.
* **AI Object Detection:** See a list of all objects the **DETR-ResNet-50** model can identify in your image.
* **AI Sentiment Analysis:** Instantly analyze the "vibe" (Positive/Negative) of the generated caption.
* **Multi-Language Translation:** Translate any caption into English, Hindi, Spanish, French, or German using `deep-translator`.
* **Text-to-Speech:** Listen to the translated caption in its native language using `gTTS`.
* **Interactive UI:** A polished, modern UI built with Streamlit, custom CSS, and a Lottie animation.

---

## 🛠️ Tech Stack

* **Front-End:** Streamlit
* **AI/ML:** PyTorch, Hugging Face `transformers`
* **AI Models:**
    * `Salesforce/blip-image-captioning-large` (Captioning)
    * `facebook/detr-resnet-50` (Object Detection)
    * `distilbert-base-uncased-finetuned-sst-2-english` (Sentiment Analysis)
* **Utilities:** `deep-translator`, `gTTS`, `streamlit-lottie`, `Pillow`

---

## 🏃 How to Run This Project Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Karma-tic/AI-Image-Caption-Generator.git](https://github.com/Karma-tic/AI-Image-Caption-Generator.git)
    cd AI-Image-Caption-Generator
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
You can view the final, working code in the project's Colab Notebook.
https://colab.research.google.com/github/Karma-tic/AI-Image-Caption-Generator/blob/main/model.ipynb
