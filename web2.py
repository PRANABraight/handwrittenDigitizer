import streamlit as st
import cv2
import numpy as np
import pytesseract
import easyocr
from PIL import Image
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
nltk.download("punkt")

# Set Tesseract path (Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# EasyOCR Reader (preload for faster performance)
easyocr_reader = easyocr.Reader(["en"])  # You can add other languages here

# Streamlit UI
st.title("📝 Handwriting Digitizer & Text Analyzer")

# Upload Image
uploaded_file = st.file_uploader("Upload a handwritten image", type=["png", "jpg", "jpeg"])

# OCR Selection
ocr_choice = st.radio("Choose OCR Engine:", ["Tesseract OCR", "EasyOCR"])

if uploaded_file:
    try:
        # Convert image to OpenCV format
        image = Image.open(uploaded_file)
        img_np = np.array(image)

        # Convert to grayscale
        if len(img_np.shape) == 2:
            gray = img_np
        elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Apply thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Display original and processed images
        st.image([image, thresh], caption=["Original Image", "Processed Image"], use_container_width=True)

        # OCR Extraction
        extracted_text = ""
        if ocr_choice == "Tesseract OCR":
            custom_config = r'--oem 3 --psm 6'
            extracted_text = pytesseract.image_to_string(thresh, config=custom_config).strip()
        elif ocr_choice == "EasyOCR":
            results = easyocr_reader.readtext(img_np, detail=0)
            extracted_text = " ".join(results)

        # Show extracted text
        st.subheader("📄 Extracted Text:")
        st.text_area("Text Output", extracted_text, height=200)

        # **TEXT ANALYSIS**
        if extracted_text:
            words = word_tokenize(extracted_text)
            sentences = sent_tokenize(extracted_text)
            words_filtered = [word.lower() for word in words if word.isalnum()]
            word_freq = Counter(words_filtered)
            most_common_words = word_freq.most_common(10)  # Top 10 words
            char_freq = Counter(extracted_text.replace(" ", ""))  # Character frequency
            
            # Display Text Analysis
            st.subheader("🔠Text Analysis")
            col1, col2, col3, col4 = st.columns(4)
            col1.write(f"**Total Characters:** {len(extracted_text)}")
            col2.write(f"**Total Words:** {len(words_filtered)}")
            col3.write(f"**Unique Words:** {len(set(words_filtered))}")
            col4.write(f"**Total Sentences:** {len(sentences)}")

            ## **Visualizations**
            st.subheader("📊 Data Visualizations")

            col1, col2 = st.columns(2)
            # Word Frequency Bar Chart
            st.write("#### Most Common Words")
            fig, ax = plt.subplots()
            sns.barplot(x=[word[0] for word in most_common_words], y=[word[1] for word in most_common_words], palette="viridis", ax=ax)
            ax.set_xlabel("Words")
            ax.set_ylabel("Frequency")
            ax.set_title("Top 10 Most Common Words")
            st.pyplot(fig)

            # Character Distribution Pie Chart
            col1.write("#### Character Frequency")
            fig, ax = plt.subplots()
            labels, values = zip(*char_freq.most_common(5))  # Top 5 characters
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("pastel"))
            ax.set_title("Top 5 Character Frequency")
            col1.pyplot(fig)

            # Sentence Length Distribution
            sentence_lengths = [len(sent.split()) for sent in sentences]
            col2.write("#### Sentence Length")
            fig, ax = plt.subplots()
            sns.histplot(sentence_lengths, bins=10, kde=True, color="blue", ax=ax)
            ax.set_xlabel("Words per Sentence")
            ax.set_ylabel("Frequency")
            ax.set_title("Sentence Length Distribution")
            col2.pyplot(fig)

        # Option to download extracted text
        st.download_button("Download Text", extracted_text, file_name="extracted_text.txt")

    except pytesseract.TesseractError as e:
        st.error(f"Tesseract OCR Error: {e}")

st.sidebar.info("Make sure the handwriting is clear and legible for better accuracy.")
