import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import streamlit as st
from models.text_generation import generate_explanation
from langchain_groq import ChatGroq
from utils.history import save_chat_history_json, load_chat_history_json, generate_summary
from models.pdf_generation import generate_pdf
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema import HumanMessage
from streamlit_mic_recorder import mic_recorder
from models.audio_handler import transcribe_audio
from models.text_to_speech import text_to_speech_pyttsx3
from utils.llm_chains import load_normal_chain
from quiz_generation.quiz import display_quiz_page
from models.pdf_handler import generate_pdf_response,collections
from models.pdf_to_quiz import process_pdf_for_quiz,ensure_models_loaded
from dotenv import load_dotenv

load_dotenv()

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="LearnFlex: Personalized Learning Assistant", layout="wide")

with open("config.yaml") as f:
    config = yaml.safe_load(f)
    
def load_chain(chat_history):
    return load_normal_chain(chat_history)

def clear_input_field():
    st.session_state.user_qa = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def toggle_pdf_chat():
    st.session_state.pdf_chat = True

def toggle_quiz():
    st.session_state.quiz = st.session_state.quiz_toggle  # Sync toggle state

def index_tracker():
    if st.session_state.session_key != "new_session":
        st.session_state.history = load_chat_history_json(os.path.join(config["sessions_dir"], st.session_state.session_key))
    else:
        st.session_state.history = []

def save_chat_history():
    """Save chat history with a summary-based filename."""
    if st.session_state.history:
        if st.session_state.session_key == "new_session":
            llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.7)
            summary = generate_summary(st.session_state.history, llm)
            filename = summary
            st.session_state.new_session_key = filename
            save_chat_history_json(st.session_state.history, os.path.join(config["sessions_dir"], filename))
        else:
            save_chat_history_json(st.session_state.history, os.path.join(config["sessions_dir"], st.session_state.session_key))


def generate_quiz_from_session():
    if st.session_state.session_key == "new_session":
        st.write("Saving the session before generating the quiz...")
        save_chat_history()
        st.session_state.session_key = st.session_state.new_session_key

    
    session_file_path = os.path.join(config["sessions_dir"], st.session_state.session_key)
    if not os.path.exists(session_file_path):
        st.error("No valid session file found for quiz generation.")
        return

    try:
        chat_history = load_chat_history_json(session_file_path)
        if not chat_history:
            st.error("No chat history found in the selected session for quiz generation.")
            return
        topic = st.session_state.topic
        display_quiz_page(chat_history,topic)
    except Exception as e:
        st.error(f"Error loading session file: {str(e)}")



# def generate_quiz_from_session():
#     """Generates a quiz from either chat history or uploaded PDF."""
#     if st.session_state.session_key == "new_session":
#         st.write("💾 Saving session before quiz generation...")
#         save_chat_history()
#         st.session_state["current_session_key"] = st.session_state.new_session_key

#     chat_history = None
#     topic = None
#     ensure_models_loaded()

#     # ✅ Handle PDF quiz generation
#     if st.session_state.get("pdf_chat", False) and "uploaded_pdf" in st.session_state:
#         st.write("✅ PDF Chat Mode Enabled")

#         try:
#             summary, topics = process_pdf_for_quiz(st.session_state.uploaded_pdf)

#             if not summary or summary.strip() == "":
#                 raise ValueError("❌ Summarized text is empty. Check PDF content.")

#             chat_history = [HumanMessage(content=summary)]
#             topic = st.session_state.get("pdf_name", "Uploaded PDF")

#         except Exception as e:
#             st.error(f"❌ Error processing PDF: {str(e)}")
#             return

#     # ✅ Handle Chat-based quiz generation
#     else:
#         session_key = st.session_state.get("current_session_key", st.session_state.session_key)
#         session_file_path = os.path.join(config["sessions_dir"], session_key)

#         if os.path.exists(session_file_path):
#             try:
#                 chat_history = load_chat_history_json(session_file_path)
#                 if not chat_history:
#                     st.error("❌ No chat history found for quiz generation.")
#                     return
#                 topic = st.session_state.topic
#             except Exception as e:
#                 st.error(f"❌ Error loading session file: {str(e)}")
#                 return
#         else:
#             st.error("❌ No valid session file found for quiz generation.")
#             return

#     # ✅ Pass extracted chat history to quiz generator
#     display_quiz_page(chat_history, topic)

# def generate_quiz_from_session():
#     if st.session_state.session_key == "new_session":
#         st.write("Saving the session before generating the quiz...")
#         save_chat_history()
#         st.session_state["current_session_key"] = st.session_state.new_session_key  # Use a different variable instead

#     chat_history = None
#     topic = None

#     # If a PDF is uploaded, extract text and use it for quiz generation
#     if st.session_state.get("pdf_chat", False) and "uploaded_pdf" in st.session_state:
#         st.write("✅ PDF Chat Mode Enabled")
#         st.write("Uploaded PDF Details:", st.session_state.uploaded_pdf)

#         try:
#             pdf_text, meta_data = process_pdf(st.session_state.uploaded_pdf)  # Extract PDF text
#             st.write("✅ Extracted PDF Text:", pdf_text[:500])  # Show first 500 chars to verify text

#             summarized_text = iterative_summarization(pdf_text)  # Summarize large text in chunks
#             st.write("✅ Summarized Text:", summarized_text[:500])  # Show first 500 chars

#             if not summarized_text or summarized_text.strip() == "":
#                 raise ValueError("❌ Summarized text is empty. Check PDF content.")

#             chat_history = [HumanMessage(content=summarized_text)]  # Treat PDF content as a message
#             topic = st.session_state.get("pdf_name", "Uploaded PDF")  # Use PDF name as the topic
        
#         except Exception as e:
#             st.error(f"❌ Error processing PDF: {str(e)}")
#             return

#     # If no PDF, use chat history
#     else:
#         session_key = st.session_state.get("current_session_key", st.session_state.session_key)  # Use new variable
#         session_file_path = os.path.join(config["sessions_dir"], session_key)

#         if os.path.exists(session_file_path):
#             try:
#                 chat_history = load_chat_history_json(session_file_path)
#                 if not chat_history:
#                     st.error("❌ No chat history found for quiz generation.")
#                     return
#                 topic = st.session_state.topic
#             except Exception as e:
#                 st.error(f"❌ Error loading session file: {str(e)}")
#                 return
#         else:
#             st.error("❌ No valid session file found for quiz generation.")
#             return

#     # ✅ Final debug before calling quiz function
#     st.write("✅ Final Chat History Before Quiz:", chat_history)
#     st.write("✅ Final Topic:", topic)

#     # ✅ Pass extracted chat history (either from PDF or chat) to display_quiz_page()
#     display_quiz_page(chat_history, topic)


def initialize_session():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
    if "history" not in st.session_state:
        st.session_state["history"] = [] 
    
    if "pdf_generated" not in st.session_state:
        st.session_state.pdf_generated = False 
        
    if "generated_answer" not in st.session_state:
        st.session_state.generated_answer = False  
        
    if "quiz" not in st.session_state:
        st.session_state.quiz = False 
    
    if "quiz_toggle" not in st.session_state:
        st.session_state.quiz_toggle = False    
    
    if "uploaded_pdf" not in st.session_state:
        st.session_state.uploaded_pdf = None
        
    if "pdf_summary" not in st.session_state:
        st.session_state.pdf_summary = None
        
    if "pdf_topics" not in st.session_state:
        st.session_state.pdf_topics = None
        
    # if "current_session_key" not in st.session_state:
    #     st.session_state.current_session_key = None
    
    if "collection" not in st.session_state:
        st.session_state.collection = {}
        
    if "send_input" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.send_input = False
        st.session_state.user_qa = ""
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
        st.session_state.generated_answer = False
        st.session_state.pdf_generated = False

def main():
    st.title("LearnFlex: Personalized Learning Assistant")
    
    chat_container = st.container()
    
    st.sidebar.title("Sessions")

    chat_sessions = ["new_session"]+os.listdir(config["sessions_dir"])
    
    initialize_session()
        
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None


    index = chat_sessions.index(st.session_state.session_index_tracker)
    
    st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index, on_change=index_tracker)
        
    if st.session_state.session_key != "new_session":
        st.session_state.history = load_chat_history_json(config["sessions_dir"] + st.session_state.session_key)
    else:
        st.session_state.history = []
    
    chat_history = StreamlitChatMessageHistory(key="history")
    
    
    if not groq_api_key:
        st.error("Error: GROQ_API_KEY is missing. Please add it to your .env file.")
        return


    model_options = ["llama-3.3-70b-versatile", "gemma2-9b-it","deepseek-r1-distill-qwen-32b"]
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = model_options[0]

    st.session_state.model_name = st.sidebar.selectbox("Choose your model", model_options)

    if st.session_state.session_key != st.session_state.session_index_tracker:
        st.session_state.quiz_toggle = False 
        st.session_state.quiz = False
        st.session_state.session_index_tracker = st.session_state.session_key 

    if chat_history.messages or st.session_state.get("pdf_chat", False): 
        st.sidebar.toggle(
            "TAKE QUIZ", 
            key="quiz_toggle", 
            value=st.session_state.quiz_toggle, 
            on_change=toggle_quiz
        )
    else:
        st.session_state.quiz = False  # Ensure quiz is off if no chat history

        
    llm = ChatGroq(api_key=groq_api_key, model_name=st.session_state.model_name, temperature=0.7)


    st.sidebar.title("Options")
    
    st.session_state.topic = st.sidebar.text_input("Enter topic", "")
    st.session_state.difficulty = st.sidebar.radio("📊 Select Difficulty Levels:", ["Easy", "Medium", "Hard"], horizontal=True,index=1)
    
    st.sidebar.title("PDF Options")
    
    st.sidebar.toggle("PDF Chat", key="pdf_chat", value=False)
    uploaded_pdf = st.sidebar.file_uploader("Upload a pdf file", accept_multiple_files=True, key="pdf_upload", type=["pdf"], on_change=toggle_pdf_chat)
    st.session_state.pdf_name = st.sidebar.text_input("Enter PDF Name", "")
    st.session_state.unit_number = st.sidebar.text_input("Enter Unit Number", "")
    st.session_state.faculty_name = st.sidebar.text_input("Enter Faculty Name", "")
    

    
    
    user_input = st.text_input("Enter your message:", key="user_input",on_change=set_send_input)
    voice_recording_col,submit_col = st.columns(2)
    
    with voice_recording_col:
        voice_recording=mic_recorder(start_prompt="Start recording",stop_prompt="Stop recording", just_once=True)
        
    with submit_col:
        submit = st.button("Submit", key="submit", help="Click to send your input")

    # Store uploaded PDF in session state
    # if "uploaded_pdf" not in st.session_state or st.session_state.uploaded_pdf is None:
    #     st.session_state.uploaded_pdf = {}

    # if uploaded_pdf:  
    #     for pdf in uploaded_pdf:  # ✅ Loop through multiple uploaded files
    #         pdf_name = pdf.name  # Extract filename
            
    #         # 🔥 Check if the PDF is already processed
    #         if pdf_name not in st.session_state.uploaded_pdf:
    #             st.session_state.uploaded_pdf[pdf_name] = {"file": pdf, "collection": None}

    #             # ✅ Pass a single PDF file instead of the entire list
    #             pdf_collection = collections(pdf)  
    #             st.session_state.uploaded_pdf[pdf_name]["collection"] = pdf_collection
    #             st.success(f"📄 {pdf_name} uploaded and processed successfully!")

    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        st.write(transcribed_audio)
        llm_chain = load_chain(chat_history)
        llm_chain.run(transcribed_audio)
    
    if uploaded_pdf is not None:
        st.session_state.uploaded_pdf = uploaded_pdf

    if st.session_state.uploaded_pdf:
        collections(st.session_state.uploaded_pdf)
        
        
    # import tempfile
    # if uploaded_pdf:
    #     if isinstance(uploaded_pdf, list):  # ✅ Ensure it's a single file
    #         uploaded_pdf = uploaded_pdf[0]  # Extract the first file

    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
    #         temp_pdf.write(uploaded_pdf.read())  
    #         temp_pdf.close()

    #         # ✅ Store the correct file path
    #         st.session_state.uploaded_pdf = temp_pdf.name  

    #         st.write(f"✅ PDF saved at: {st.session_state.uploaded_pdf}")

    #         # ✅ Call the `collections()` function with valid file paths
    #         collections(st.session_state.uploaded_pdf)  # ✅ Use correct path

    if st.session_state.get("quiz", False):  # If quiz toggle is ON
        st.write("Generating a quiz based on the selected chat session...")
        generate_quiz_from_session()
    
        
    if submit or st.session_state.get("send_input", False): 
        with st.spinner("Processing your message..."):
            user_message = st.session_state.user_qa
            
            if user_message:
                if st.session_state.get("pdf_chat", False) and "uploaded_pdf" in st.session_state:                
                    llm_response = generate_pdf_response(user_message, llm)
                else:
                    llm_response = generate_explanation(
                        user_message, 
                        st.session_state.difficulty, 
                        llm, 
                        st.session_state.topic
                    )

                # Update chat history
                chat_history.add_user_message(user_message)
                chat_history.add_ai_message(llm_response)
                st.session_state.generated_answer = True
                st.session_state.pdf_generated = False 

        st.session_state.send_input = False


    if chat_history.messages:
        with chat_container:
            st.write("Chat History")
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content)

    save_chat_history() 


    if st.sidebar.button("Generate PDF"):
        if "pdf_path" not in st.session_state or st.session_state.pdf_generated is False:
            with st.spinner("Generating PDF..."):
                st.session_state["pdf_path"] = generate_pdf(
                    st.session_state.history,
                    st.session_state.topic,
                    st.session_state.unit_number,
                    st.session_state.faculty_name,
                    st.session_state.pdf_name,
                    generate_summary(st.session_state.history, llm)
                )
                st.session_state.pdf_generated = True 
                st.sidebar.success("PDF has been generated.")

        if st.session_state.get("pdf_path"):
            with open(st.session_state["pdf_path"], "rb") as pdf_file:
                st.sidebar.download_button(
                    label="Download PDF",
                    data=pdf_file,
                    file_name=st.session_state["pdf_path"],
                    mime="application/pdf"
                )
        else:
            st.error("Failed to generate or locate the PDF.")


    st.sidebar.title("Audio options")
    chat_number = st.sidebar.text_input("Enter chat number (or 'all' for full conversation):")

    if not chat_number:
        chat_number = 'all'
    if st.sidebar.button("Convert to Speech"):
        full_text = None 

        if chat_number.lower() == 'all':
            if len(st.session_state.history) > 0:
                full_text = ""
                for i in range(0, len(st.session_state.history), 2):
                    user_msg = st.session_state.history[i]
                    if i + 1 < len(st.session_state.history):
                        assistant_msg = st.session_state.history[i + 1]
                        full_text += f"Question: {user_msg.content} .... Answer: {assistant_msg.content} "
            else:
                st.error("No conversation history available to convert!")
        else:
            try:
                chat_index = 2*(int(chat_number) -1)  # Convert to 0-based index
                if 0 <= chat_index < len(st.session_state.history):
                    selected_chat = st.session_state.history[chat_index]
                    full_text = f"Question: {selected_chat.content} " 
                    if chat_index + 1 < len(st.session_state.history):
                        assistant_response = st.session_state.history[chat_index + 1]
                        full_text += f"Answer: {assistant_response.content}"
                    
                else:
                    st.error(f"Invalid chat number! Please enter a number between 1 and {len(st.session_state.history)}.")
            except ValueError:
                st.error("Please enter a valid number or 'all' for full conversation.")

        if full_text:
            audio_file = text_to_speech_pyttsx3(full_text)

            st.sidebar.audio(audio_file, format="audio/mp3")

            st.sidebar.download_button(
                label="Download Audio",
                data=audio_file,
                file_name="selected_chat.mp3" if chat_number.lower() != 'all' else "full_conversation.mp3",
                mime="audio/mp3"
            )
            
            
if __name__ == "__main__":
    main()