import os
import faiss
import tempfile
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key="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")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embedding_dim = 768


# def process_pdf(uploaded_pdf):
#     """
#     Processes a single uploaded PDF: Extracts text and metadata.
#     Uses OCR for scanned PDFs (images inside PDFs).
#     """

#     # ✅ If uploaded_pdf is a list, extract the first file
#     if isinstance(uploaded_pdf, list):
#         uploaded_pdf = uploaded_pdf[0]

#     # ✅ Handle both Streamlit UploadedFile and raw bytes
#     if isinstance(uploaded_pdf, bytes):
#         uploaded_pdf = io.BytesIO(uploaded_pdf)  # Convert bytes to file-like object

#     # ✅ Ensure the file is readable
#     uploaded_pdf.seek(0)

#     # ✅ Extract text and metadata
#     all_text = []
#     metadata = []

#     reader = PdfReader(uploaded_pdf)
#     for page_num, page in enumerate(reader.pages):
#         text = page.extract_text()

#         if text and text.strip():  # ✅ If text exists, use it directly
#             all_text.append(text)
#             metadata.append({"source": uploaded_pdf.name, "page": page_num})

#         else:
#             st.write(f"🔍 No text found on Page {page_num + 1}, applying OCR...")

#             # Convert PDF page to an image
#             images = convert_from_path(uploaded_pdf, first_page=page_num+1, last_page=page_num+1)
#             for img in images:
#                 text = pytesseract.image_to_string(img)
#                 all_text.append(text)
#                 metadata.append({"source": uploaded_pdf.name, "page": page_num, "ocr": True})

#     return all_text, metadata


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


def chunk_text(text_list, metadata_list):

    # text_splitter = SemanticChunker(embeddings)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=500, separators=["\n\n", ".", "?", "!", ";"]
    )
    chunks = []
    chunk_metadata = []

    for text, meta in zip(text_list, metadata_list):
        chunked_texts = text_splitter.create_documents([text])
        chunks.extend(chunked_texts)
        chunk_metadata.extend([meta] * len(chunked_texts))
    st.write(f"✅ Chunking completed: {len(chunks)} chunks created.")
    return chunks, chunk_metadata


# def chunk_text(text_list, metadata_list):
#     text_splitter = SemanticChunker(embeddings)

#     chunks = []
#     chunk_metadata = []

#     for text, meta in zip(text_list, metadata_list):
#         chunked_texts = text_splitter.split_text(text)

#         merged_chunks = []
#         temp_chunk = ""
#         for chunk in chunked_texts:
#             temp_chunk += " " + chunk
#             if len(temp_chunk) > 500:  # Adjust for better chunk size
#                 merged_chunks.append(temp_chunk)
#                 temp_chunk = ""
#         if temp_chunk:
#             merged_chunks.append(temp_chunk)

#         chunks.extend(merged_chunks)
#         chunk_metadata.extend([meta] * len(merged_chunks))

#     return chunks, chunk_metadata

# def rewrite_query(user_query):
#     query_prompt = PromptTemplate(
#         template="""
#         Given the following query, **rephrase it** to improve retrieval from a **vector database**.  
#         1️⃣ Expand keywords to include relevant synonyms.  
#         2️⃣ Maintain original meaning.  
#         3️⃣ Format it as a standalone, complete question.

#         ❓ **Original Query**:
#         {query}

#         🔎 **Optimized Query**:
#         """,
#         input_variables=["query"]
#     )

#     query_chain = LLMChain(llm=llm, prompt=query_prompt)
#     rewritten_query = query_chain.run({"query": user_query})

#     print(f"✅ Original Query: {user_query}")
#     print(f"🔄 Optimized Query: {rewritten_query}")

#     return rewritten_query



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


def create_faiss_index(text_chunks):
    chunk_embeddings = np.array([embeddings.embed_query(chunk.page_content) for chunk in text_chunks], dtype=np.float32)
    faiss.normalize_L2(chunk_embeddings)  # ✅ Normalize before adding
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(chunk_embeddings)
    return index




def retrieve_relevant_chunk(query, index, text_chunks, metadata):
    query_embedding = np.array([embeddings.embed_query(query)], dtype=np.float32)
    D, I = index.search(query_embedding, k=5)  # Get top 5 matches

    if I is None or len(I[0]) == 0 or I[0][0] == -1:
        st.write("⚠️ No matching chunks found for query.")
        return []

    retrieved_data = []
    for i in I[0]:
        if 0 <= i < len(text_chunks):  # Ensure valid index
            retrieved_data.append((text_chunks[i].page_content, metadata[i]))  # Proper tuple
    st.write(f"📄 **Retrieved Data Format:** {retrieved_data}")

    # Debugging
    st.write(f"🔍 Retrieved {len(retrieved_data)} chunks:")
    for text, data in retrieved_data:
        st.write(f"📄 **Page {data.get('page', 'N/A')}:** {text[:200]}...")  # Show sample text

    return retrieved_data



def loading_data(uploaded_pdf):
    if "index" in st.session_state:
        st.info("📂 PDF is already loaded. No need to reload.")
        return

    with st.spinner("🔄 Loading and processing PDF..."):
        pdf_text, metadata = process_pdf(uploaded_pdf)
        st.write("pdf_text = " , pdf_text)
        text_chunks, chunk_metadata = chunk_text(pdf_text, metadata)
        st.write("text_chunks = " , text_chunks)
        index = create_faiss_index(text_chunks)

        st.session_state.index = index
        st.session_state.processed_chunks = text_chunks
        st.session_state.chunk_metadata = chunk_metadata

    st.success("✅ PDF successfully processed and indexed!")


def get_vectors_from_db(pdf_hash):
    mongo_uri = os.getenv('MONGO_URI')
    client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
    db = client["learn_flex"]
    collection = db["pdf_rag"]

    # ✅ Check if vectors exist
    existing_vectors = collection.find({"pdf_hash": pdf_hash})
    vector_count = collection.count_documents({"pdf_hash": pdf_hash})
    
    if vector_count > 0:
        st.write(f"📄 Found {vector_count} chunks in the database for {pdf_hash}.")

        # ✅ Ensure proper vector search object is returned
        return MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index",
            relevance_score_fn="cosine",
        )

    st.write(f"⚠️ No stored vectors found for {pdf_hash}.")
    return None
    
# def get_vectors_from_db(pdf_hash):
#     mongo_uri = os.getenv('MONGO_URI')
#     client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
#     db = client["learn_flex"]
#     collection = db["pdf_rag"]

#     # ✅ Check if the PDF exists in the database
#     existing_vectors = collection.find_one({"pdf_hash": pdf_hash})
    
#     if existing_vectors:
#         print(f"📄 Found vectors in the database for {pdf_hash}.")
        
#         # ✅ Return a MongoDBAtlasVectorSearch object instead of a list
#         return MongoDBAtlasVectorSearch(
#             collection=collection,
#             embedding=embeddings,
#             index_name="vector_index",
#             relevance_score_fn="cosine",
#         )

#     print(f"⚠️ No stored vectors found for {pdf_hash}.")
#     return None

def save_pdf_hash_to_db(pdf_hash):
    """Store PDF hash in MongoDB to avoid redundant processing."""
    mongo_uri = os.getenv('MONGO_URI')
    client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
    db = client["learn_flex"]
    collection = db["pdf_rag"]

    if not collection.find_one({"pdf_hash": pdf_hash}):
        collection.insert_one({"pdf_hash": pdf_hash})
        st.write("✅ PDF hash stored successfully.")
    else:
        st.write("⚠️ PDF hash already exists in the database.")


def create_mongo_index(text_chunks, metadata,pdf_hash):
    mongo_uri = os.getenv('MONGO_URI')
    if not mongo_uri:
        st.write("❌ ERROR: MongoDB URI is missing!")
        return None  # 🚨 Ensure function does not return None

    client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
    db = client["learn_flex"]
    collection = db["pdf_rag"]
    # pdf_hash = metadata[0]["source"]  # Use filename as the identifier

    # ✅ Check if PDF already exists
    existing_chunks = collection.count_documents({"pdf_hash": pdf_hash})
    if existing_chunks > 0:
        st.write("✅ PDF already exists in vector database. Skipping insertion...")

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
        st.write(f"✅ Stored {len(documents)} chunks in MongoDB.")

    try:
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index",
            relevance_score_fn="cosine",
        )
        st.write("✅ Created MongoDB Vector Store:", vector_store)
        return vector_store
    except Exception as e:
        st.write(f"❌ ERROR: Failed to create vector store - {str(e)}")
        return None  # 🚨 Ensure function does not return None


def get_pdf_hash(pdf_file):
    """Generates a unique hash for a given PDF file."""
    st.write(f"DEBUG: pdf_file type = {type(pdf_file)}")

    if isinstance(pdf_file, dict):
        st.write(f"DEBUG: Dictionary keys received: {list(pdf_file.keys())}") 
        if "file" in pdf_file:
            pdf_file = pdf_file["file"]
        elif "content" in pdf_file:  
            pdf_file = io.BytesIO(pdf_file["content"])
        else:
            st.write("⚠️ Warning: Unexpected dictionary format.")
            return None  

    if isinstance(pdf_file, bytes):
        pdf_file = io.BytesIO(pdf_file)  

    if not hasattr(pdf_file, "seek"):
        st.write(f"⚠️ Warning: Invalid file format: {type(pdf_file)}")
        return None  

    hasher = hashlib.md5()
    pdf_file.seek(0)
    hasher.update(pdf_file.read())
    pdf_file.seek(0)  

    pdf_hash = hasher.hexdigest()
    st.write(f"✅ Generated PDF Hash: {pdf_hash}")
    return pdf_hash

def collections(uploaded_pdf):
    """Handles PDF processing and ensures only the latest uploaded PDF is used for retrieval."""

    if "uploaded_pdfs" not in st.session_state:
        st.session_state.uploaded_pdfs = {}

    with st.spinner("Loading Documents..."):
        pdf_hash = get_pdf_hash(uploaded_pdf)
        if not pdf_hash:
            st.write("❌ ERROR: Could not generate PDF hash!")
            return

        st.write(f"DEBUG: Processing PDF Hash {pdf_hash}")

        # ✅ Store the latest uploaded PDF hash
        st.session_state.latest_pdf_hash = pdf_hash  # 🚀 Store the latest PDF hash only

        # ✅ Store the file in session (prevents re-uploading)
        pdf_name = uploaded_pdf.name  
        st.session_state.uploaded_pdfs[pdf_name] = {"file": uploaded_pdf, "pdf_hash": pdf_hash}

        # ✅ Check if vectors already exist in MongoDB
        stored_collection = get_vectors_from_db(pdf_hash)

        if stored_collection:
            st.write(f"✅ PDF already processed. Using stored vectors for {pdf_hash}...")
            st.session_state.collection = stored_collection
        else:
            st.write(f"⚡ Processing new PDF ({pdf_hash}) and storing vectors...")
            pdf_text, metadata = process_pdf(uploaded_pdf)  
            text_chunks, chunk_metadata = chunk_text(pdf_text, metadata)
            collection = create_mongo_index(text_chunks, chunk_metadata, pdf_hash)

            # ✅ Store the collection for this session
            st.session_state.collection = collection
            save_pdf_hash_to_db(pdf_hash)  # Store hash in MongoDB

    st.success(f"✅ {pdf_name} uploaded and processed successfully!")


# def retrieve_relevant_chunk(query):
#     """Retrieves relevant chunks only from the latest uploaded PDF's vector store."""
    
#     pdf_hash = st.session_state.latest_pdf_hash
#     vectorstore = get_vectors_from_db(pdf_hash)

#     if not pdf_hash:
#         st.write("❌ No PDF uploaded. Please upload a document first.")
#         return []

#     if not vectorstore:
#         st.write(f"❌ No stored vectors found for PDF Hash: {pdf_hash}")
#         return []

#     st.write(f"✅ Using PDF Hash {pdf_hash} for retrieval.")

#     # Perform similarity search using only the latest uploaded PDF
#     results = vectorstore.similarity_search(query, k=5) 

#     if not results:
#         st.write("⚠️ No relevant chunks retrieved.")
#         return []
    
#     st.write(f"✅ Retrieved {len(results)} relevant chunks:")
#     for idx, res in enumerate(results):
#         st.write(f"🔹 Chunk {idx+1}: {res.page_content}...")  # Show first 300 chars

#     retrieved_data = [(res.page_content, res.metadata) for res in results]
#     return retrieved_data


# def retrieve_relevant_chunk(query):
#     """Retrieves relevant chunks only from the latest uploaded PDF's vector store."""

#     # ✅ Get the latest uploaded PDF hash
#     pdf_hash = st.session_state.latest_pdf_hash
#     vectorstore = get_vectors_from_db(pdf_hash)
#     # pdf_hash = st.session_state.get("latest_pdf_hash")
#     if not pdf_hash:
#         st.write("❌ No PDF uploaded. Please upload a document first.")
#         return []

#     st.write(f"DEBUG: Using PDF Hash {pdf_hash} for retrieval.")


#     if not vectorstore:
#         st.write(f"❌ No stored vectors found for PDF Hash: {pdf_hash}")
#         return []

#     st.write(f"✅ Found vectorstore for PDF Hash: {pdf_hash}")

#     # ✅ Perform similarity search using only the latest uploaded PDF
#     results = vectorstore.similarity_search(query, k=5) 
#     if not results:
#         st.write("⚠️ No relevant chunks retrieved.")
#     else:
#         st.write(f"✅ Retrieved {len(results)} relevant chunks.")

#     retrieved_data = [(res.page_content, res.metadata) for res in results]
#     return retrieved_data
def generate_pdf_response(user_message, llm):
        # ✅ Ensure FAISS index is loaded
    if "index" not in st.session_state:
        return "❌ No PDF data available. Please upload a document first."

    optimized_query = rewrite_query(user_message) or user_message 
    # retrieved_data = retrieve_relevant_chunk(optimized_query)
    retrieved_data = retrieve_relevant_chunk(
                optimized_query, 
                st.session_state.index, 
                st.session_state.processed_chunks, 
                st.session_state.chunk_metadata,
                # st.session_state.collection
            )
    if not retrieved_data:
        return "⚠️ No relevant information found in the PDF."

    # 🔥 🔥 NEW: Show only the most relevant chunks 🔥 🔥
    top_context = "\n\n".join([
        f"📖 **Source:** {data.get('source', 'Unknown')}, **Page:** {data.get('page', 'N/A')}\n{text}"
        for text, data in retrieved_data[:3]  # 🔥 Only use the top 3 chunks!
    ])


    prompt_template = PromptTemplate(
        template="""
        You are an AI assistant specialized in retrieving **factual and precise** answers from the provided document context.  
        Your response must be **strictly based** on the given information.  

        ### 🔹 Guidelines:
        - **Extract only relevant details** from the provided context.  
        - Keep responses **clear, concise, and within 100 words**.  
        - If the context **does not contain the answer**, respond with:  
        **"The provided context does not contain sufficient information to answer this question."**  
        - **DO NOT** add external knowledge, assumptions, or opinions.  

        ### 📜 Context:
        {context}

        ### ❓ User Question:
        {question}

        ### ✅ Answer:
        """,
        input_variables=["context", "question"]
    )


    chain = prompt_template | llm
    llm_response = chain.invoke({"context": top_context, "question": user_message})
    return llm_response


# def generate_pdf_response(user_message, llm):
#     optimized_query = rewrite_query(user_message)

#     retrieved_data = retrieve_relevant_chunk(optimized_query)

#     if not retrieved_data:
#         return "⚠️ No relevant information found in the PDF."

#     # Improve prompt for better LLM response
#     context = "\n\n".join([
#         f"📖 **Source:** {data.get('source', 'Unknown')}, **Page:** {data.get('page', 'N/A')}\n{text}"
#         for text, data in retrieved_data
#     ])

#     prompt_template = PromptTemplate(
#         template="""
#         You are an advanced AI assistant answering based on a document.
#         Follow these guidelines:
#         1️⃣ **Read the context carefully** and extract the key information.
#         2️⃣ **Provide a detailed, structured answer** with clear explanations.
#         3️⃣ **Summarize concisely** if needed but retain important details.
        
#         📜 **Context:**
#         {context}

#         ❓ **Question:**
#         {question}

#         ✅ **Answer:**
#         """,
#         input_variables=["context", "question"]
#     )

#     chain = prompt_template | llm
#     llm_response = chain.invoke({"context": context, "question": user_message})
#     return llm_response
