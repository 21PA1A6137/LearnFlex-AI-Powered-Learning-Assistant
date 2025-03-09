import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.memory import StreamlitChatMessageHistory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from PyPDF2 import PdfReader
from pymongo import MongoClient
import ssl
import hashlib
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_mongodb import MongoDBAtlasVectorSearch



llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key="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")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embedding_dim = 768

# def process_pdf(uploaded_pdfs):
#     """
#     Processes uploaded PDFs: Extracts text, chunks it, stores embeddings in MongoDB.
#     """
#     all_text = []
#     metadata = []


#     for pdf in uploaded_pdfs:
#         pdf_name = pdf.name if hasattr(pdf, "name") else pdf.get("name", "Unknown.pdf")  # ✅ Handles both cases
        
#         reader = PdfReader(pdf)
#         for page_num, page in enumerate(reader.pages):
#             text = page.extract_text()
#             if text:
#                 all_text.append(text)
#                 metadata.append({"source": pdf_name, "page": page_num})

#     return all_text, metadata

import io
from PyPDF2 import PdfReader

def process_pdf(uploaded_pdf):
    """
    Processes a single uploaded PDF: Extracts text and metadata.
    """

    # ✅ Handle both Streamlit UploadedFile and raw bytes
    if isinstance(uploaded_pdf, bytes):
        uploaded_pdf = io.BytesIO(uploaded_pdf)  # Convert bytes to file-like object

    # ✅ Handle both `UploadedFile` and dictionary formats
    if hasattr(uploaded_pdf, "name"):  
        pdf_name = uploaded_pdf.name
    elif isinstance(uploaded_pdf, dict) and "name" in uploaded_pdf:
        pdf_name = uploaded_pdf["name"]
    else:
        pdf_name = "Unknown.pdf"

    print(f"DEBUG: Processing PDF - {pdf_name}, Type: {type(uploaded_pdf)}")

    # ✅ Ensure the file is readable
    uploaded_pdf.seek(0)

    # ✅ Extract text and metadata
    all_text = []
    metadata = []

    reader = PdfReader(uploaded_pdf)
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            all_text.append(text)
            metadata.append({"source": pdf_name, "page": page_num})

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

def chunk_text(text_list, metadata_list):

    text_splitter = SemanticChunker(embeddings)
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

# def get_vectors_from_db(pdf_hash):
#     """Retrieve stored vector data for a given PDF hash from MongoDB."""
#     mongo_uri = os.getenv('MONGO_URI')
#     client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
#     db = client["learn_flex"]
#     collection = db["pdf_rag"]

#     # Check if vectors exist for the given hash
#     existing_vectors = list(collection.find({"pdf_hash": pdf_hash}))
    
#     if existing_vectors:
#         print("📄 Found existing vectors in the database.")
#         return MongoDBAtlasVectorSearch(
#             collection=collection,
#             embedding=embeddings,
#             index_name="vector_index",
#             relevance_score_fn="cosine",
#         )
    
#     return None  # No vectors found, needs processing

def get_vectors_from_db(pdf_hash):
    """Retrieve stored vector data for a given PDF hash from MongoDB."""
    mongo_uri = os.getenv('MONGO_URI')
    client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
    db = client["learn_flex"]
    collection = db["pdf_rag"]

    # Retrieve all chunks related to this PDF
    existing_vectors = list(collection.find({"pdf_hash": pdf_hash}))

    if existing_vectors:
        print(f"📄 Found {len(existing_vectors)} chunks in the database for {pdf_hash}.")
        return existing_vectors  # Return actual vector data instead of a search index

    print("⚠️ No vectors found in the database.")
    return None  # No vectors found, needs processing



def save_pdf_hash_to_db(pdf_hash):
    """Store PDF hash in MongoDB to avoid redundant processing."""
    mongo_uri = os.getenv('MONGO_URI')
    client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
    db = client["learn_flex"]
    collection = db["pdf_rag"]

    if not collection.find_one({"pdf_hash": pdf_hash}):
        collection.insert_one({"pdf_hash": pdf_hash})
        print("✅ PDF hash stored successfully.")
    else:
        print("⚠️ PDF hash already exists in the database.")

def create_mongo_index(text_chunks, metadata):
    mongo_uri = os.getenv('MONGO_URI')
    client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
    db = client["learn_flex"]
    collection = db["pdf_rag"]

    pdf_hash = metadata[0]["source"]  # Use filename as the identifier

    # Check if data already exists
    existing_chunks = collection.count_documents({"pdf_hash": pdf_hash})
    if existing_chunks > 0:
        print("✅ PDF already exists in vector database. Skipping insertion...")
        return

    # Store chunks with their embeddings
    documents = []
    for chunk, meta in zip(text_chunks, metadata):
        embedding = embeddings.embed_query(chunk)
        document = {
            "pdf_hash": pdf_hash,
            "text": chunk,
            "embedding": embedding,  # Store embeddings for search
            "metadata": meta
        }
        documents.append(document)

    if documents:
        collection.insert_many(documents)
        print(f"✅ Stored {len(documents)} chunks in MongoDB.")




# def create_mongo_index(text_chunks, metadata):
#     mongo_uri = os.getenv('MONGO_URI')
#     client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
#     db = client["learn_flex"]
#     collection = db["pdf_rag"]
    
#     # Check if the PDF hash is already stored in MongoDB
#     pdf_hash = metadata[0]["source"]  # Assume first metadata contains the filename
#     existing_hash = collection.find_one({"pdf_hash": pdf_hash})

#     if existing_hash:
#         print("✅ PDF already exists in vector database. Skipping insertion...")
#         return MongoDBAtlasVectorSearch(
#             collection=collection,
#             embedding=embeddings,
#             index_name="vector_index",
#             relevance_score_fn="cosine",
#         )

#     # If not found, process the new document
#     documents = []
#     for chunk, meta in zip(text_chunks, metadata):
#         embedding = embeddings.embed_query(chunk)
#         document = {
#             "pdf_hash": pdf_hash,  # Store hash for quick lookup
#             "text": chunk,
#             "embedding": embedding,
#             "metadata": meta
#         }
#         documents.append(document)
    
#     if documents:
#         collection.insert_many(documents)

#     return MongoDBAtlasVectorSearch(
#         collection=collection,
#         embedding=embeddings,
#         index_name="vector_index",
#         relevance_score_fn="cosine",
#     )



# def create_mongo_index(text_chunks, metadata):
#     mongo_uri = os.getenv('MONGO_URI')
#     client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
#     db = client["learn_flex"]
#     collection = db["pdf_rag"]
    
#     documents = []
#     for chunk, meta in zip(text_chunks, metadata):
#         embedding = embeddings.embed_query(chunk)
#         document = {
#             "text": chunk,
#             "embedding": embedding,
#             "metadata": meta
#         }
#         documents.append(document)
    
#     if documents:
#         collection.insert_many(documents)

#     vector_store = MongoDBAtlasVectorSearch(
#         collection=collection,
#         embedding=embeddings,
#         index_name="vector_index",
#         relevance_score_fn="cosine",
#     )
#     return vector_store


# def retrieve_relevant_chunk(query,vectorstore):
#     """Uses MongoDB Vector Search to retrieve relevant case studies."""
#     results = vectorstore.similarity_search(query, k=5) 
#     retrieved_data = []
#     # print(results) #
#     for res in results:
#         # print(res)
#         retrieved_data.append((res.page_content, res.metadata))
#     # print(retrieved_data)
#     return retrieved_data


# def get_pdf_hash(pdf_file):
#     hasher = hashlib.md5()

#     if isinstance(pdf_file, str):
#         raise TypeError("Expected a file-like object, but received a string.")

#     pdf_file.seek(0)  # ✅ Ensure file starts from beginning
#     hasher.update(pdf_file.read())  
#     pdf_file.seek(0)  # ✅ Reset pointer after reading

#     return hasher.hexdigest()

# import hashlib
# import io  # ✅ Import io to handle bytes data

# def get_pdf_hash(pdf_file):
#     """Generates a unique hash for a given PDF file."""
    
#     # ✅ Debugging statement
#     print(f"DEBUG: pdf_file type = {type(pdf_file)}")

#     # ✅ If `pdf_file` is a dictionary, extract the actual file
#     if isinstance(pdf_file, dict):
#         if "file" in pdf_file:
#             pdf_file = pdf_file["file"]  # ✅ Extract file object
#         elif "content" in pdf_file:  
#             pdf_file = io.BytesIO(pdf_file["content"])  # ✅ Convert bytes to file-like object
#         else:
#             raise ValueError("Invalid PDF structure. Expected 'file' or 'content' key in dictionary.")

#     # ✅ Ensure `pdf_file` is a file-like object
#     if not hasattr(pdf_file, "seek"):
#         raise TypeError("Expected a file-like object but got: " + str(type(pdf_file)))

#     hasher = hashlib.md5()
#     pdf_file.seek(0)  # ✅ Ensure reading from the beginning
#     hasher.update(pdf_file.read())
#     pdf_file.seek(0)  # ✅ Reset file pointer after reading

#     return hasher.hexdigest()

import hashlib
import io

def get_pdf_hash(pdf_file):
    """Generates a unique hash for a given PDF file."""

    # ✅ Debugging statement
    print(f"DEBUG: pdf_file type = {type(pdf_file)}")

    # ✅ If `pdf_file` is a dictionary, extract the actual file
    if isinstance(pdf_file, dict):
        print(f"DEBUG: Dictionary keys received: {list(pdf_file.keys())}")  # Add this debug line
        if "file" in pdf_file:
            pdf_file = pdf_file["file"]  # ✅ Extract file object
        elif "content" in pdf_file:  
            pdf_file = io.BytesIO(pdf_file["content"])  # ✅ Convert bytes to file-like object
        else:
            print("⚠️ Warning: Unexpected dictionary format. Expected 'file' or 'content' key.")
            return None  # Return None instead of raising an error

    # ✅ Convert bytes to file-like object if necessary
    if isinstance(pdf_file, bytes):
        pdf_file = io.BytesIO(pdf_file)  # ✅ Convert bytes to file-like object

    # ✅ Ensure `pdf_file` is a file-like object
    if not hasattr(pdf_file, "seek"):
        print(f"⚠️ Warning: Invalid file format. Received {type(pdf_file)}")
        return None  # Return None instead of raising an error

    # ✅ Compute hash
    hasher = hashlib.md5()
    pdf_file.seek(0)  # ✅ Ensure reading from the beginning
    hasher.update(pdf_file.read())
    pdf_file.seek(0)  # ✅ Reset file pointer after reading

    return hasher.hexdigest()



def collections(uploaded_pdf):
    if "collection" not in st.session_state:
        st.session_state.collection = {}
    
    if "uploaded_pdf" not in st.session_state:
        st.session_state.uploaded_pdf = uploaded_pdf  # ✅ Store the uploaded file

    with st.spinner("Loading Documents..."):
        print(f"DEBUG: pdf_file type = {type(uploaded_pdf)}, keys = {uploaded_pdf.keys() if isinstance(uploaded_pdf, dict) else 'N/A'}")

        pdf_hash = get_pdf_hash(uploaded_pdf)

        # Check if vectors already exist in MongoDB
        stored_collection = get_vectors_from_db(pdf_hash)
        
        if stored_collection:
            st.write("✅ PDF already processed. Loading stored vectors...")
            st.session_state.collection[pdf_hash] = stored_collection
        else:
            st.write("⚡ Processing new PDF and storing vectors...")
            pdf_text, metadata = process_pdf(uploaded_pdf)  # Process single PDF
            text_chunks, chunk_metadata = chunk_text(pdf_text, metadata)
            collection = create_mongo_index(text_chunks, chunk_metadata)

            # Save collection and hash
            st.session_state.collection[pdf_hash] = collection
            save_pdf_hash_to_db(pdf_hash)  # Store hash to prevent redundant processing
    st.success("Documents loaded successfully!")


# def collections(uploaded_pdf):
#     if "collection" not in st.session_state:
#         st.session_state.collection = {}

#     with st.spinner("Loading Documents..."):
#         print(f"DEBUG: pdf_file type = {type(uploaded_pdf)}, keys = {uploaded_pdf.keys() if isinstance(uploaded_pdf, dict) else 'N/A'}")

#         pdf_hash = get_pdf_hash(uploaded_pdf)
#         print(f"✅ Generated PDF Hash: {pdf_hash}")

#         # Check if vectors already exist in MongoDB
#         stored_collection = get_vectors_from_db(pdf_hash)
        
#         if stored_collection:
#             st.write("✅ PDF already processed. Loading stored vectors...")
#             st.session_state.collection[pdf_hash] = stored_collection
#         else:
#             st.write("⚡ Processing new PDF and storing vectors...")

#             pdf_text, metadata = process_pdf(uploaded_pdf)  # Extract text
#             print(f"DEBUG: Extracted text (first 200 chars): {pdf_text[:200]}")

#             text_chunks, chunk_metadata = chunk_text(pdf_text, metadata)  # Chunk the text
#             print(f"✅ Created {len(text_chunks)} text chunks.")

#             collection = create_mongo_index(text_chunks, chunk_metadata)  # Store in MongoDB
#             print(f"✅ Stored {len(text_chunks)} chunks in MongoDB.")

#             # Save collection in session state
#             st.session_state.collection[pdf_hash] = collection
#             save_pdf_hash_to_db(pdf_hash)  # Store hash to avoid reprocessing
#             print(f"✅ Stored PDF hash in database: {pdf_hash}")

#     st.success("Documents loaded successfully!")

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def retrieve_relevant_chunk(query, uploaded_pdf):
    """Retrieves relevant chunks from MongoDB Vector Store."""

    if "uploaded_pdfs" not in st.session_state or st.session_state.uploaded_pdfs is None:
        print("⚠️ No uploaded PDF found in session state. Please upload a PDF first.")
        return []

    uploaded_pdf = st.session_state.uploaded_pdfs
    pdf_hash = get_pdf_hash(uploaded_pdf)

    stored_chunks = get_vectors_from_db(pdf_hash)  # Fetch stored embeddings
    
    if not stored_chunks:
        print(f"⚠️ No stored vectors found for PDF Hash: {pdf_hash}")
        return []

    # Compute embedding for the user query
    query_embedding = embeddings.embed_query(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    # Compare with stored embeddings
    similarities = []
    for chunk in stored_chunks:
        chunk_embedding = np.array(chunk["embedding"]).reshape(1, -1)
        similarity = cosine_similarity(query_embedding, chunk_embedding)[0][0]
        similarities.append((similarity, chunk["text"], chunk["metadata"]))

    # Sort chunks by similarity score
    similarities.sort(reverse=True, key=lambda x: x[0])

    # Return top-k most relevant chunks
    top_chunks = similarities[:5]  # Retrieve top 5 matches
    retrieved_data = [(res[1], res[2]) for res in top_chunks]

    return retrieved_data





# def retrieve_relevant_chunk(query, uploaded_pdf):
#     """Retrieves relevant chunks from MongoDB Vector Store."""

#     print(f"DEBUG: uploaded_pdf type = {type(uploaded_pdf)}")

#     if isinstance(uploaded_pdf, list) and uploaded_pdf:
#         pdf_file = uploaded_pdf[0]
#     else:
#         pdf_file = uploaded_pdf  

#     if isinstance(pdf_file, dict):
#         pdf_file = pdf_file.get("file") or io.BytesIO(pdf_file.get("content", b""))  

#     if not hasattr(pdf_file, "read"):
#         raise TypeError(f"Expected a file-like object but got: {type(pdf_file)}")

#     pdf_hash = get_pdf_hash(pdf_file)
#     print(f"✅ Retrieved PDF Hash: {pdf_hash}")

#     vectorstore = st.session_state.collection.get(pdf_hash)  

#     if not vectorstore:
#         print(f"⚠️ No stored vectors found for PDF Hash: {pdf_hash}")
#         print(f"DEBUG: Available keys in session state: {list(st.session_state.collection.keys())}")
#         return []

#     print(f"✅ Found vectorstore for PDF Hash: {pdf_hash}")

#     results = vectorstore.similarity_search(query, k=5) 
#     if not results:
#         print("⚠️ No relevant chunks retrieved.")
#     else:
#         print(f"✅ Retrieved {len(results)} relevant chunks.")

#     retrieved_data = [(res.page_content, res.metadata) for res in results]

#     return retrieved_data




# def collections(uploaded_pdf):
#     if "collection" not in st.session_state:
#         with st.spinner("Loading Documents..."):
#             pdf_text, metadata = process_pdf(uploaded_pdf)
#             text_chunks, chunk_metadata = chunk_text(pdf_text, metadata)
#             collection = create_mongo_index(text_chunks, chunk_metadata)

#             st.session_state.collection = collection
#         st.success("Documents loaded successfully!")

def generate_pdf_response(user_message,llm):

    optimized_query = rewrite_query(user_message)
    if "uploaded_pdfs" not in st.session_state or not st.session_state.uploaded_pdfs:
        return "❌ No PDF uploaded. Please upload a document first."

    uploaded_pdf = st.session_state.uploaded_pdfs
    retrieved_data = retrieve_relevant_chunk(
        optimized_query, 
        uploaded_pdf
    )

    if not retrieved_data:
        llm_response = "No relevant information found in the PDF."

    context = "\n\n".join([
        f"Source: {data.get('source', 'Unknown')}, Page {data.get('page', 'N/A')}\n{text}"
        for text, data in retrieved_data
    ])
    prompt_template = PromptTemplate(
        template="""
        You are an AI assistant answering based on a Document provided.
        Use the context below to provide a precise response.
        Context: {context}
        Question: {question}
        Answer:
        """,
        input_variables=["context", "question"]
    )
    chain = prompt_template | llm
    llm_response = chain.invoke({"context": context, "question": user_message})
    return llm_response

#     submit = st.button("Submit", key="submit", help="Click to send your input")
#     if submit or st.session_state.get("send_input", False):
#         with st.spinner("Processing your message..."):
#             user_message = st.session_state.user_qa
            
            
#             optimized_query = rewrite_query(user_message)
#             retrieved_data = retrieve_relevant_chunk(
#                 optimized_query, 
#                 st.session_state.collection
#             )

#             if not retrieved_data:
#                 st.warning("No relevant case study found.")
#                 return

#             context = "\n\n".join([
#                 f"Source: {data.get('source', 'Unknown')}, Page {data.get('page', 'N/A')}\n{text}"
#                 for text, data in retrieved_data
#             ])
            
#             prompt_template = PromptTemplate(
#                 template="""
#                 You are an expert legal assistant answering case study questions.
#                 Use the retrieved context below to provide a precise response.
#                 If the context is not sufficient, say 'I don't know.'
                
#                 Context: {context}
#                 Question: {question}
#                 Answer:
#                 """,
#                 input_variables=["context", "question"]
#             )

#             if user_message:
#                 chain = LLMChain(llm=llm, prompt=prompt_template)
#                 answer = chain.run({"context": context, "question": user_message})
#                 st.session_state.chat_history.add_user_message(user_message)
#                 st.session_state.chat_history.add_ai_message(answer)

#             st.session_state.send_input = False

#     if st.session_state.chat_history.messages:
#         with st.container():
#             st.write("🗂️ Chat History")
#             for message in st.session_state.chat_history.messages:
#                 st.chat_message(message.type).write(message.content)


# if __name__ == "__main__":
#     main()