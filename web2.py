import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from spellchecker import SpellChecker


# Download NLTK data
nltk.download("punkt")

# Initialize EasyOCR Reader
easyocr_reader = easyocr.Reader(["en"])

# Initialize SpellChecker
spell = SpellChecker()

# Streamlit UI
st.title("📝 Handwriting Digitizer & Text Analyzer")

# Sidebar
st.sidebar.header("Settings")
ocr_choice = st.sidebar.radio("Choose OCR Engine:", ["EasyOCR", "Google Vision API"])

st.info("For best accuracy, ensure clear handwriting on a plain white background.")

# Upload Image
uploaded_file = st.file_uploader("Upload a handwritten image", type=["png", "jpg", "jpeg"])

if uploaded_file:
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

    # Display images
    st.image([image, thresh], caption=["Original Image", "Processed Image"], use_container_width=True)

    extracted_text = ""

    # **OCR Selection**
    results = easyocr_reader.readtext(img_np, detail=0)
    extracted_text = " ".join(results)

    # **Spell Checking**
    corrected_text = []
    misspelled_words = spell.unknown(word_tokenize(extracted_text))
    for word in word_tokenize(extracted_text):
        if word in misspelled_words:
            corrected_word = spell.correction(word)
            corrected_text.append(corrected_word)
        else:
            corrected_text.append(word)
    corrected_text = " ".join(corrected_text)

    # Display extracted text
    st.subheader("📄 Extracted Text:")
    st.text_area("Original Text Output", extracted_text, height=100)
    st.text_area("Corrected Text Output", corrected_text, height=100)

    # **TEXT ANALYSIS**
    if corrected_text:
        words = word_tokenize(corrected_text)
        sentences = sent_tokenize(corrected_text)
        words_filtered = [word.lower() for word in words if word.isalnum()]
        word_freq = Counter(words_filtered)
        most_common_words = word_freq.most_common(10)  # Top 10 words
        char_freq = Counter(corrected_text.replace(" ", ""))  # Character frequency

        # Display Text Analysis
        st.subheader("🔠 Text Analysis")
        col1, col2, col3, col4 = st.columns(4)
        col1.write(f"**Total Characters:** {len(corrected_text)}")
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
    st.download_button("Download Corrected Text", corrected_text, file_name="corrected_text.txt")
