import streamlit as st
import os
from pdf2image import convert_from_path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from keybert import KeyBERT
from models.pdf_handler import process_pdf

# ✅ Ensure CUDA is used if available
device = "cuda" if torch.cuda.is_available() else "cpu"


# ✅ Define model paths to avoid redownloading
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")


# ✅ STEP 1: Extract Text from PDFs (Both Text-Based & Scanned)
# def extract_text_from_pdf(pdf_path):
#     """Extracts text from PDFs using PyMuPDF for text-based and Tesseract for scanned PDFs."""
#     extracted_text = []
#     doc = fitz.open(pdf_path)

#     for page in doc:
#         text = page.get_text("text")
#         if text.strip():
#             extracted_text.append(text)

#     if not extracted_text:
#         st.write("🔍 No text found! Using OCR...")
#         with tempfile.TemporaryDirectory() as temp_dir:
#             images = convert_from_path(pdf_path, output_folder=temp_dir, fmt="jpeg")
#             for img in images:
#                 text = pytesseract.image_to_string(img)
#                 extracted_text.append(text)

#     return "\n".join(extracted_text)

# ✅ STEP 2: Summarize Extracted Text

# ✅ Ensure models are loaded before calling `iterative_summarization()`
@st.cache_resource()
def load_models():
    global tokenizer, summarization_model, nlp, kw_model
    try:
        st.write("🔄 Loading models... (only once)")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True)
        summarization_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True).to(device)

        nlp = spacy.load("en_core_web_sm")
        kw_model = KeyBERT()
        
        st.write("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")

def ensure_models_loaded():
    """Loads models only if they haven't been initialized yet."""
    if "tokenizer" not in globals():
        load_models()

            
def iterative_summarization(text, chunk_size=2000, max_summary_length=600):
    """Summarizes long text by chunking and summarizing iteratively."""

    # ✅ If already summarized, return cached result
    if "pdf_summary" in st.session_state:
        return st.session_state.pdf_summary

    if isinstance(text, list):
        text = " ".join(text)

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []

    for i, chunk in enumerate(chunks):
        st.write(f"⚡ Summarizing chunk {i+1}/{len(chunks)}...")
        inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True).to(device)
        summary_ids = summarization_model.generate(inputs.input_ids, max_length=max_summary_length, min_length=100)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    final_summary = " ".join(summaries)

    # ✅ Store summary in `st.session_state` to avoid re-running
    st.session_state.pdf_summary = final_summary

    if len(final_summary.split()) > chunk_size:
        st.write("⚡ Recursively summarizing the combined content...")
        return iterative_summarization(final_summary, chunk_size, max_summary_length)
    return final_summary



# ✅ STEP 3: Extract Key Topics from Summary
def extract_keywords_tfidf(text, top_n=5):
    """Extracts key topics using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    keywords = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [kw[0] for kw in keywords]

def extract_entities_spacy(text):
    """Extracts key topics using Named Entity Recognition (NER)."""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT", "EVENT"]]
    return list(set(entities))

def extract_keywords_bert(text, top_n=5):
    """Extracts key topics using BERT-based embeddings."""
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n)
    return [kw[0] for kw in keywords]

# ✅ STEP 4: Full Pipeline (Extract → Summarize → Find Topics → Generate Quiz)
def process_pdf_for_quiz(pdf_path,pdf_text):
    """Processes a PDF to extract text, summarize it, and find key topics."""

    # ✅ Check if we already processed this PDF
    if "pdf_summary" in st.session_state and st.session_state.get("pdf_path") == pdf_path:
        st.write("✅ Using cached PDF summary and topics...")
        return st.session_state.pdf_summary, st.session_state.pdf_topics

    st.write("📂 Extracting text from PDF...")
    # extracted_text, meta_data = process_pdf(pdf_path)
    st.session_state.extracted_text = pdf_text  # ✅ Store extracted text

    st.write("📄 Summarizing text...")
    summary = iterative_summarization(pdf_text)
    st.session_state.pdf_summary = summary  # ✅ Store summary

    st.write("🔍 Extracting key topics...")
    tfidf_keywords = extract_keywords_tfidf(summary)
    spacy_entities = extract_entities_spacy(summary)
    bert_keywords = extract_keywords_bert(summary)

    # ✅ Combine results
    key_topics = list(set(tfidf_keywords + spacy_entities + bert_keywords))
    st.session_state.pdf_topics = key_topics  # ✅ Store topics

    st.write(f"✅ Key Topics Identified: {key_topics}")

    return summary, key_topics

