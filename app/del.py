# import os
# from pymongo import MongoClient
# from dotenv import load_dotenv
# load_dotenv()
# mongo_uri = os.getenv('MONGO_URI')

# client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
# db = client["learn_flex"]
# collection = db["pdf_rag"]

# # res = collection.find({})
# # print(res)
# collection.delete_many({})
# print("successfully deleted")

# import torch
# print("CUDA Available:", torch.cuda.is_available())
# print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected!")

# import streamlit as st
# import fitz  # PyMuPDF for text PDFs
# import pytesseract  # OCR for scanned PDFs
# from pdf2image import convert_from_path  # Convert PDF to images
# import tempfile
# import os
# import torch
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# import spacy
# from keybert import KeyBERT

# # ✅ Check for CUDA availability
# device = "cuda" if torch.cuda.is_available() else "cpu"
# st.write(f"Using device: {device.upper()}")

# # ✅ Load models safely
# def load_models():
#     """Loads necessary models for summarization and keyword extraction."""
#     global tokenizer, summarization_model, nlp, kw_model

#     try:
#         # ✅ Load Summarization Model
#         summarizer_model = "sshleifer/distilbart-cnn-12-6"
#         tokenizer = AutoTokenizer.from_pretrained(summarizer_model)
#         summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model).to(device)

#         # ✅ Load NLP Models
#         nlp = spacy.load("en_core_web_sm")
#         kw_model = KeyBERT()
#     except Exception as e:
#         st.error(f"❌ Error loading models: {e}")

# # ✅ Run this only once
# if __name__ == "__main__":
#     load_models()

# # ✅ STEP 1: Extract Text from PDFs (Both Text-Based & Scanned)
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

# # ✅ STEP 2: Summarize Extracted Text
# def iterative_summarization(text, chunk_size=2000, max_summary_length=600):
#     """Summarizes long text by chunking and summarizing iteratively."""
#     if isinstance(text, list):
#         text = " ".join(text)

#     chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
#     summaries = []

#     for i, chunk in enumerate(chunks):
#         st.write(f"⚡ Summarizing chunk {i+1}/{len(chunks)}...")
#         inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True).to(device)
#         summary_ids = summarization_model.generate(inputs.input_ids, max_length=max_summary_length, min_length=100)
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         summaries.append(summary)

#     combined_summary = " ".join(summaries)

#     if len(combined_summary.split()) > chunk_size:
#         st.write("⚡ Recursively summarizing the combined content...")
#         return iterative_summarization(combined_summary, chunk_size, max_summary_length)

#     return combined_summary

# # ✅ STEP 3: Extract Key Topics from Summary
# def extract_keywords_tfidf(text, top_n=5):
#     """Extracts key topics using TF-IDF."""
#     vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
#     tfidf_matrix = vectorizer.fit_transform([text])
#     feature_names = vectorizer.get_feature_names_out()
#     scores = tfidf_matrix.toarray()[0]

#     keywords = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)[:top_n]
#     return [kw[0] for kw in keywords]

# def extract_entities_spacy(text):
#     """Extracts key topics using Named Entity Recognition (NER)."""
#     doc = nlp(text)
#     entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT", "EVENT"]]
#     return list(set(entities))

# def extract_keywords_bert(text, top_n=5):
#     """Extracts key topics using BERT-based embeddings."""
#     keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n)
#     return [kw[0] for kw in keywords]

# # ✅ STEP 4: Full Pipeline (Extract → Summarize → Find Topics)
# def process_pdf_for_quiz(pdf_path):
#     """Processes a PDF to extract text, summarize it, and find key topics."""
#     st.write("📂 Extracting text from PDF...")
#     extracted_text = extract_text_from_pdf(pdf_path)

#     st.write("📄 Summarizing text...")
#     summary = iterative_summarization(extracted_text)

#     st.write("🔍 Extracting key topics...")
#     tfidf_keywords = extract_keywords_tfidf(summary)
#     st.write(f"TF-IDF Keywords: {tfidf_keywords}")
#     spacy_entities = extract_entities_spacy(summary)
#     st.write(f"SpaCy Entities: {spacy_entities}")
#     bert_keywords = extract_keywords_bert(summary)
#     st.write(f"BERT Keywords: {bert_keywords}")

#     # ✅ Combine results
#     key_topics = list(set(tfidf_keywords + spacy_entities + bert_keywords))
#     st.write(f"✅ Key Topics Identified: {key_topics}")

#     return summary, key_topics

# # ✅ User Uploads PDF
# pdf_path = st.file_uploader("Upload a PDF", type=["pdf"])
# if pdf_path:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
#         temp_pdf.write(pdf_path.read())  # Save uploaded file to temp storage
#         temp_pdf.close()
#         st.write("Processing uploaded PDF...")
#         summary, topics = process_pdf_for_quiz(temp_pdf.name)
        
#         st.write("\n📌 **Final Summary:**\n", summary)
#         st.write("\n🔹 **Key Topics:**\n", topics)
