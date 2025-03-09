import os
import tempfile
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.memory import StreamlitChatMessageHistory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from PyPDF2 import PdfReader
from pymongo import MongoClient
import io
import ssl
import hashlib
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_mongodb import MongoDBAtlasVectorSearch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

groq_api_key = os.getenv("GRQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)




def process_pdf(uploaded_pdf):
    """
    Processes a single uploaded PDF: Extracts text and metadata.
    Uses OCR for scanned PDFs (images inside PDFs).
    """

    if isinstance(uploaded_pdf, list):  
        uploaded_pdf = uploaded_pdf[0]  # ✅ Extract the first file  

    # ✅ Convert Streamlit file-like object to a readable format
    pdf_stream = io.BytesIO(uploaded_pdf.read())

    # ✅ Extract text and handle encrypted PDFs
    all_text = []
    metadata = []
    reader = PdfReader(pdf_stream)

    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception as e:
            st.write(f"🔒 Encrypted PDF detected on Page {page_num + 1}. Unable to extract text.")
            continue  # Skip if text extraction fails

        if text and text.strip():  
            all_text.append(text)
            metadata.append({"source": uploaded_pdf.name, "page": page_num})
        else:
            st.write(f"🔍 No text found on Page {page_num + 1}, applying OCR...")

            # ✅ Convert PDF page to an image for OCR
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(pdf_stream.getbuffer())  # ✅ Save to a temp file
                temp_pdf.close()

                images = convert_from_path(temp_pdf.name, first_page=page_num+1, last_page=page_num+1)

                for img in images:
                    text = pytesseract.image_to_string(img)
                    all_text.append(text)
                    metadata.append({"source": uploaded_pdf.name, "page": page_num, "ocr": True})

                os.remove(temp_pdf.name)  # ✅ Cleanup temp file

    return all_text, metadata


def get_pdf_names(folder_path):
    pdf_list = []
    index = 1
    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            index = str(index)
            pdf_list.append(index +" "+ file)
            index = int(index)+1
    return pdf_list

def chunk_text(text_list, metadata_list,embeddings):

    text_splitter = SemanticChunker(embeddings,breakpoint_threshold_type="gradient")
    chunks = []
    chunk_metadata = []

    for text, meta in zip(text_list, metadata_list):
        chunked_texts = text_splitter.split_text(text)
        chunks.extend(chunked_texts)
        chunk_metadata.extend([meta] * len(chunked_texts))

    return chunks, chunk_metadata

def rewrite_query(user_query):
    query_prompt = PromptTemplate(
        template="""
        Given the following query, refine it to be more structured, clear
        while maintaining the original meaning.
        Question: {query}
        Optimized Query:
        """,
        input_variables=["query"]
    )

    query_chain = LLMChain(llm=llm, prompt=query_prompt)
    return query_chain.run({"query": user_query})


def create_mongo_index(text_chunks, metadata,embeddings):
    mongo_uri = os.getenv('MONGO_URI')
    if not mongo_uri:
        st.write("❌ ERROR: MongoDB URI is missing!")
        return None  # 🚨 Ensure function does not return None

    client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
    db = client["learn_flex"]
    collection = db["pdf_rag"]
    pdf_hash = metadata[0]["source"]  # Use filename as the identifier

    # ✅ Check if PDF already exists
    existing_chunks = collection.count_documents({"pdf_hash": pdf_hash})
    # if existing_chunks > 0:
    #     st.write("✅ PDF already exists in vector database. Skipping insertion...")

    documents = []
    for chunk, meta in zip(text_chunks, metadata):
        embedding = embeddings.embed_query(chunk)
        document = {
            "pdf_hash": pdf_hash,
            "text": chunk,
            "embedding": embedding,
            "metadata": meta
        }
        documents.append(document)
    
    if documents:
        collection.insert_many(documents)
        # st.write(f"✅ Stored {len(documents)} chunks in MongoDB.")

    try:
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index",
            relevance_score_fn="cosine",
        )
        # st.write("✅ Created MongoDB Vector Store:", vector_store)
        return vector_store
    except Exception as e:
        st.write(f"❌ ERROR: Failed to create vector store - {str(e)}")
        return None  # 🚨 Ensure function does not return None


def collections(uploaded_pdf):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embedding_dim = 768
    if "collection" not in st.session_state or not st.session_state.collection:
        with st.spinner("Loading Documents..."):
            pdf_text, metadata = process_pdf(uploaded_pdf)
            st.write("Extracted Text:", pdf_text)
            text_chunks, chunk_metadata = chunk_text(pdf_text, metadata,embeddings)
            st.write("Chuncked Text:", text_chunks)
            collection = create_mongo_index(text_chunks, metadata,embeddings)
            st.session_state.collection = collection
        return pdf_text
    st.success("📄 Document loaded successfully!")



def retrieve_relevant_chunk(query, vectorstore):
    """Uses MongoDB Vector Search to retrieve relevant case studies."""
    
    if vectorstore is None:
        st.write("❌ ERROR: Vectorstore is None! Check MongoDB connection.")
        return []
    results = vectorstore.similarity_search(query, k=5) 
    retrieved_data = []
    for res in results:
        retrieved_data.append((res.page_content, res.metadata))
    # st.write("retrieved_data",retrieved_data)
    return retrieved_data


def generate_pdf_response(user_message,llm):

    optimized_query = rewrite_query(user_message)
    retrieved_data = retrieve_relevant_chunk(
        optimized_query,st.session_state.collection 
        # uploaded_pdf
    )
    st.write("Retrieved Data:", retrieved_data)

    if not retrieved_data:
        llm_response = "No relevant information found in the PDF."

    context = "\n\n".join([
        f"Source: {data.get('source', 'Unknown')}, Page {data.get('page', 'N/A')}\n{text}"
        for text, data in retrieved_data
    ])
    # st.write("PDF Response:", context)


    prompt_template = PromptTemplate(
        template="""
        You are an intelligent AI assistant that provides **accurate, well-structured, and detailed responses** 
        based on the given document. Your goal is to **answer precisely, concisely, and clearly** while ensuring 
        that **all key details are included**.

        🔹 **Context (Extracted from Document)**:
        {context}
        
        🔹 **User's Question**:
        {question}

        🔹 **Your Answer**:
        - 📌 **Ensure accuracy** by using only the provided context.
        - 🔍 **Explain in a structured way** (bullets, numbered points, or short paragraphs).
        - 📖 **Provide examples or references** from the document when relevant.
        - ❌ **Avoid speculation** if the answer is not in the document.
        
        Now, provide a **detailed yet concise answer**:
        """,
        input_variables=["context", "question"]
    )

    chain = prompt_template | llm
    llm_response = chain.invoke({"context": context, "question": user_message})
    return llm_response