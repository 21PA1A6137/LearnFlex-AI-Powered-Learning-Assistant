# import streamlit as st
# from models.text_generation import generate_explanation
# # from models.text_to_speech import text_to_speech
# # from app.pdf_generator import generate_pdf
# # from app.quiz import generate_quiz
# # from app.user_progress import update_user_score
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# load_dotenv()


# def main():
#     groq_api_key = os.getenv("GROQ_API_KEY")
    
#     llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0)
#     topic = st.text_input("Enter Topic")
#     difficulty = st.sidebar.selectbox("Select Difficulty", ["easy", "medium", "hard"])


#     if st.button("Generate Explanation"):
#         explanation = generate_explanation(topic, difficulty,llm)
#         st.write(explanation)
        
#         pdf_name = f"explanation_{topic}_{difficulty}.pdf"
#         audio_name = f"explanation_{topic}_{difficulty}.mp3"
        
#         # if st.button("Download PDF"):
#         #     generate_pdf(explanation, pdf_name)
#         #     st.download_button("Download PDF", pdf_name)
        
#         # if st.button("Download Audio"):
#         #     text_to_speech(explanation, audio_name)
#         #     st.download_button("Download Audio", audio_name)

#     # Quiz Section
#     # st.subheader("Take a Quiz")
#     # quiz = generate_quiz(difficulty)
#     # score = 0
    
#     # for q in quiz:
#     #     answer = st.text_input(q['question'])
#     #     if answer == q['answer']:
#     #         score += 1
    
#     # st.write(f"Your Score: {score}")
#     # if st.button("Submit Quiz"):
#     #     update_user_score(user_id, score)

# main()








# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import yaml
# import streamlit as st
# from models.text_generation import generate_explanation
# from langchain_groq import ChatGroq
# from utils.history import save_chat_history_json, get_time_stamp, load_chat_history_json, generate_summary
# from models.pdf_generation import generate_pdf
# from langchain.memory import StreamlitChatMessageHistory
# from streamlit_mic_recorder import mic_recorder
# from models.audio_handler import transcribe_audio
# from models.text_to_speech import text_to_speech_pyttsx3
# from utils.llm_chains import load_normal_chain
# from dotenv import load_dotenv

# load_dotenv()

# # os.environ["LANGCHAIN_TRACING_V2"] = "true"
# # os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# groq_api_key="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"

# st.set_page_config(page_title="LearnFlex: Personalized Learning Assistant", layout="wide")


# with open("config.yaml") as f:
#     config = yaml.safe_load(f)
    
# def load_chain(chat_history):
#     return load_normal_chain(chat_history)

# def clear_input_field():
#     st.session_state.user_qa = st.session_state.user_input
#     st.session_state.user_input = ""

# # set the send input flag
# def set_send_input():
#     st.session_state.send_input = True
#     clear_input_field()

# # set the session index
# def index_tracker():
#     if st.session_state.session_key != "new_session":
#         st.session_state.history = load_chat_history_json(os.path.join(config["sessions_dir"], st.session_state.session_key))
#     else:
#         st.session_state.history = []

# def save_chat_history():
#     """Save chat history with a summary-based filename."""
#     if st.session_state.history:
#         if st.session_state.session_key == "new_session":
#             llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.7)
#             summary = generate_summary(st.session_state.history, llm)
#             filename = summary
#             st.session_state.new_session_key = filename
#             save_chat_history_json(st.session_state.history, os.path.join(config["sessions_dir"], filename))
#         else:
#             save_chat_history_json(st.session_state.history, os.path.join(config["sessions_dir"], st.session_state.session_key))


# # def full_conversation_to_speech(history):
# #     # Check the message type and extract content accordingly
# #     full_text = " ".join([msg.content for msg in history])  # Access the content directly
# #     return text_to_speech_pyttsx3(full_text)

# def main():
#     st.title("LearnFlex: Personalized Learning Assistant")
    
#     chat_container = st.container()
    
#     st.sidebar.title("Sessions")

#     chat_sessions = ["new_session"]+os.listdir(config["sessions_dir"])
    
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = []
        
#     if "history" not in st.session_state:
#         st.session_state["history"] = [] 
    
#     if "pdf_generated" not in st.session_state:
#         st.session_state.pdf_generated = False 
        
#     if "generated_answer" not in st.session_state:
#         st.session_state.generated_answer = False  

    
#     if "send_input" not in st.session_state:
#         st.session_state.session_key = "new_session"
#         st.session_state.send_input = False
#         st.session_state.user_qa = ""
#         st.session_state.new_session_key = None
#         st.session_state.session_index_tracker = "new_session"
#         st.session_state.generated_answer = False
#         st.session_state.pdf_generated = False
        
#     if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
#         st.session_state.session_index_tracker = st.session_state.new_session_key
#         st.session_state.new_session_key = None
        
#     index = chat_sessions.index(st.session_state.session_index_tracker)
#     st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index, on_change=index_tracker)

#     if st.session_state.session_key != "new_session":
#         st.session_state.history = load_chat_history_json(config["sessions_dir"] + st.session_state.session_key)
#     else:
#         st.session_state.history = []
    
    
    
#     chat_history = StreamlitChatMessageHistory(key="history")
#     # groq_api_key = os.getenv("GROQ_API_KEY")
#     if not groq_api_key:
#         st.error("Error: GROQ_API_KEY is missing. Please add it to your .env file.")
#         return

#     llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.7)

#     st.sidebar.title("Options")
#     st.session_state.subject = st.sidebar.text_input("Enter Subject", "")
#     st.session_state.difficulty = st.sidebar.selectbox("Select Difficulty", ["easy", "medium", "hard"])
#     st.session_state.pdf_name = st.sidebar.text_input("Enter PDF Name", "")
#     st.session_state.unit_number = st.sidebar.text_input("Enter Unit Number", "") # Replace with your dynamic unit number
#     st.session_state.faculty_name = st.sidebar.text_input("Enter Faculty Name", "")
    
#     user_input = st.text_input("Enter your message:", key="user_input",on_change=set_send_input)
#     voice_recording_col,submit_col = st.columns(2)
    
#     with voice_recording_col:
#         voice_recording=mic_recorder(start_prompt="Start recording",stop_prompt="Stop recording", just_once=True)
        
#     with submit_col:
#         submit = st.button("Submit", key="submit", help="Click to send your input")

    
#     if voice_recording:
#         transcribed_audio = transcribe_audio(voice_recording["bytes"])
#         print(transcribed_audio)
#         llm_chain = load_chain(chat_history)
#         llm_chain.run(transcribed_audio)
    
    
#     if submit or st.session_state.get("send_input", False): 
#         with st.spinner("Processing your message..."):
#             user_message = st.session_state.user_qa
#             if user_message:
#                 llm_response = generate_explanation(
#                     user_message, 
#                     st.session_state.difficulty, 
#                     llm, 
#                     st.session_state.subject
#                 )
#                 chat_history.add_user_message(user_message)
#                 chat_history.add_ai_message(llm_response)
                
#                 st.session_state.generated_answer = True
#                 st.session_state.pdf_generated = False 

#         st.session_state.send_input = False

#     if chat_history.messages:
#         with chat_container:
#             st.write("Chat History")
#             for message in chat_history.messages:
#                 st.chat_message(message.type).write(message.content)

#     save_chat_history() 




#     if st.sidebar.button("Generate PDF"):
#         if "pdf_path" not in st.session_state or st.session_state.pdf_generated is False:
#             with st.spinner("Generating PDF..."):
#                 st.session_state["pdf_path"] = generate_pdf(
#                     st.session_state.history,
#                     st.session_state.subject,
#                     st.session_state.unit_number,
#                     st.session_state.faculty_name,
#                     st.session_state.pdf_name,
#                     generate_summary(st.session_state.history, llm)
#                 )
#                 st.session_state.pdf_generated = True 
#                 st.sidebar.success("PDF has been generated.")

#         if st.session_state.get("pdf_path"):
#             with open(st.session_state["pdf_path"], "rb") as pdf_file:
#                 st.sidebar.download_button(
#                     label="Download PDF",
#                     data=pdf_file,
#                     file_name=st.session_state["pdf_path"],
#                     mime="application/pdf"
#                 )
#         else:
#             st.error("Failed to generate or locate the PDF.")

#     chat_number = st.sidebar.text_input("Enter chat number (or 'all' for full conversation):")

#     if not chat_number:
#         chat_number = 'all'
#     if st.sidebar.button("Convert to Speech"):
#         full_text = None  # Default value for the text to be converted

#         if chat_number.lower() == 'all':
#             if len(st.session_state.history) > 0:
#                 full_text = ""
#                 for i in range(0, len(st.session_state.history), 2):
#                     user_msg = st.session_state.history[i]
#                     if i + 1 < len(st.session_state.history):
#                         assistant_msg = st.session_state.history[i + 1]
#                         full_text += f"Question: {user_msg.content} .... Answer: {assistant_msg.content} "
#             else:
#                 st.error("No conversation history available to convert!")
#         else:
#             try:
#                 chat_index = 2*(int(chat_number) -1)  # Convert to 0-based index
#                 if 0 <= chat_index < len(st.session_state.history):
#                     selected_chat = st.session_state.history[chat_index]
#                     full_text = f"Question: {selected_chat.content} " 
#                     if chat_index + 1 < len(st.session_state.history):
#                         assistant_response = st.session_state.history[chat_index + 1]
#                         full_text += f"Answer: {assistant_response.content}"
                    
#                 else:
#                     st.error(f"Invalid chat number! Please enter a number between 1 and {len(st.session_state.history)}.")
#             except ValueError:
#                 st.error("Please enter a valid number or 'all' for full conversation.")

#         if full_text:
#             audio_file = text_to_speech_pyttsx3(full_text)

#             st.sidebar.audio(audio_file, format="audio/mp3")

#             st.sidebar.download_button(
#                 label="Download Audio",
#                 data=audio_file,
#                 file_name="selected_chat.mp3" if chat_number.lower() != 'all' else "full_conversation.mp3",
#                 mime="audio/mp3"
#             )
            
            
# if __name__ == "__main__":
#     main()













# import sqlite3
# import os
# import sys
# import uuid
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import yaml
# import streamlit as st
# from models.text_generation import generate_explanation
# from utils.history import save_chat_history_json, get_time_stamp, load_chat_history_json, generate_summary
# from langchain_groq import ChatGroq
# from models.audio_handler import transcribe_audio
# from models.pdf_generation import generate_pdf
# from models.text_to_speech import text_to_speech_pyttsx3
# from utils.llm_chains import load_normal_chain
# from dotenv import load_dotenv

# load_dotenv()
# groq_api_key="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"


# with open("config.yaml") as f:
#     config = yaml.safe_load(f)
# # Database setup function
# def create_table():
#     conn = sqlite3.connect("chat_history.db")
#     cursor = conn.cursor()
#     cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
#                         session_key TEXT,
#                         role TEXT,
#                         message TEXT,
#                         timestamp TEXT)''')
#     conn.commit()
#     conn.close()

# # Function to save chat history to the database
# def save_chat_history_to_db(session_key, role, message, timestamp):
#     conn = sqlite3.connect("chat_history.db")
#     cursor = conn.cursor()
#     cursor.execute("INSERT INTO chat_history (session_key, role, message, timestamp) VALUES (?, ?, ?, ?)",
#                    (session_key, role, message, timestamp))
#     conn.commit()
#     conn.close()

# # Function to load chat history from the database
# def load_chat_history_from_db(session_key):
#     conn = sqlite3.connect("chat_history.db")
#     cursor = conn.cursor()
#     cursor.execute("SELECT role, message, timestamp FROM chat_history WHERE session_key=?", (session_key,))
#     rows = cursor.fetchall()
#     conn.close()
    
#     chat_history = []
#     for row in rows:
#         chat_history.append({"role": row[0], "message": row[1], "timestamp": row[2]})
#     return chat_history

# # Session management: Unique session key generation
# def generate_unique_session_key():
#     return str(uuid.uuid4())

# def clear_input_field():
#     st.session_state.user_qa = st.session_state.user_input
#     st.session_state.user_input = ""

# # set the send input flag
# def set_send_input():
#     st.session_state.send_input = True
#     clear_input_field()

# # set the session index
# def index_tracker():
#     if st.session_state.session_key != "new_session":
#         st.session_state.history = load_chat_history_from_db(st.session_state.session_key)
#     else:
#         st.session_state.history = []

# def save_chat_history():
#     if st.session_state.history:
#         try:
#             if st.session_state.session_key == "new_session":
#                 # Generate a session key when a new session starts
#                 st.session_state.session_key = generate_unique_session_key()
#                 llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.7)
#                 summary = generate_summary(st.session_state.history, llm)
#                 filename = summary
#                 st.session_state.new_session_key = filename
#             # Save each message to the database
#             for message in st.session_state.history:
#                 save_chat_history_to_db(st.session_state.session_key, message['role'], message['message'], message['timestamp'])
#         except Exception as e:
#             st.error(f"Error saving chat history: {e}")
#     else:
#         st.error("No chat history to save.")

# def main():
#     st.title("LearnFlex: Personalized Learning Assistant")
    
#     chat_container = st.container()
    
#     st.sidebar.title("Sessions")
#     chat_sessions = ["new_session"]+os.listdir(config["sessions_dir"])  # Placeholder for sessions
    
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = []
        
#     if "history" not in st.session_state:
#         st.session_state["history"] = [] 
    
#     if "pdf_generated" not in st.session_state:
#         st.session_state.pdf_generated = False 
        
#     if "generated_answer" not in st.session_state:
#         st.session_state.generated_answer = False  

#     if "send_input" not in st.session_state:
#         st.session_state.session_key = generate_unique_session_key()  # Unique session key generation
#         st.session_state.send_input = False
#         st.session_state.user_qa = ""
#         st.session_state.new_session_key = None
#         st.session_state.session_index_tracker = "new_session"
#         st.session_state.generated_answer = False
#         st.session_state.pdf_generated = False
        
#     if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
#         st.session_state.session_index_tracker = st.session_state.new_session_key
#         st.session_state.new_session_key = None
        
#     index = chat_sessions.index(st.session_state.session_index_tracker)
#     st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index, on_change=index_tracker)

#     if st.session_state.session_key != "new_session":
#         try:
#             st.session_state.history = load_chat_history_from_db(st.session_state.session_key)
#         except Exception as e:
#             st.error(f"Error loading chat history: {e}")
#     else:
#         st.session_state.history = []
    
#     # Prepare the chat history in memory
#     chat_history = [{"role": message["role"], "message": message["message"]} for message in st.session_state.history]

#     # Checking the Groq API key
#     groq_api_key = "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"
#     if not groq_api_key:
#         st.error("Error: GROQ_API_KEY is missing. Please add it to your .env file.")
#         return

#     llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.7)

#     st.sidebar.title("Options")
#     st.session_state.subject = st.sidebar.text_input("Enter Subject", "")
#     st.session_state.difficulty = st.sidebar.selectbox("Select Difficulty", ["easy", "medium", "hard"])
#     st.session_state.pdf_name = st.sidebar.text_input("Enter PDF Name", "")
#     st.session_state.unit_number = st.sidebar.text_input("Enter Unit Number", "")
#     st.session_state.faculty_name = st.sidebar.text_input("Enter Faculty Name", "")
    
#     user_input = st.text_input("Enter your message:", key="user_input", on_change=set_send_input)
#     submit = st.button("Submit", key="submit", help="Click to send your input")

#     if submit or st.session_state.get("send_input", False): 
#         with st.spinner("Processing your message..."):
#             user_message = st.session_state.user_qa
#             if user_message:
#                 llm_response = generate_explanation(
#                     user_message, 
#                     st.session_state.difficulty, 
#                     llm, 
#                     st.session_state.subject
#                 )
                
#                 # Add both user and AI messages to the chat history
#                 st.session_state.history.append({"role": "user", "message": user_message, "timestamp": get_time_stamp()})
#                 st.session_state.history.append({"role": "ai", "message": llm_response, "timestamp": get_time_stamp()})
                
#                 st.session_state.generated_answer = True
#                 st.session_state.pdf_generated = False
#         st.session_state.send_input = False

#     # Display chat history in the UI
#     if chat_history:
#         with chat_container:
#             st.write("Chat History")
#             for message in chat_history:
#                 st.chat_message(message["role"]).write(message["message"])

#     save_chat_history()  # Save chat history after every interaction

#     # PDF Generation
#     if st.sidebar.button("Generate PDF"):
#         if "pdf_path" not in st.session_state or st.session_state.pdf_generated is False:
#             with st.spinner("Generating PDF..."):
#                 try:
#                     st.session_state["pdf_path"] = generate_pdf(
#                         st.session_state.history,
#                         st.session_state.subject,
#                         st.session_state.unit_number,
#                         st.session_state.faculty_name,
#                         st.session_state.pdf_name,
#                         "Summary of the Chat History"  # Placeholder for summary
#                     )
#                     st.session_state.pdf_generated = True 
#                     st.sidebar.success("PDF has been generated.")
#                 except Exception as e:
#                     st.error(f"Error generating PDF: {e}")
        
#         if st.session_state.get("pdf_path"):
#             with open(st.session_state["pdf_path"], "rb") as pdf_file:
#                 st.sidebar.download_button(
#                     label="Download PDF",
#                     data=pdf_file,
#                     file_name=st.session_state["pdf_path"],
#                     mime="application/pdf"
#                 )
#         else:
#             st.error("Failed to generate or locate the PDF.")

#     # Convert to Speech
#     chat_number = st.sidebar.text_input("Enter chat number (or 'all' for full conversation):")
#     if not chat_number:
#         chat_number = 'all'

#     if st.sidebar.button("Convert to Speech"):
#         full_text = None  # Default value for the text to be converted
#         if chat_number.lower() == 'all':
#             if len(st.session_state.history) > 0:
#                 full_text = ""
#                 for i in range(0, len(st.session_state.history), 2):
#                     user_msg = st.session_state.history[i]
#                     if i + 1 < len(st.session_state.history):
#                         assistant_msg = st.session_state.history[i + 1]
#                         full_text += f"Question: {user_msg['message']} .... Answer: {assistant_msg['message']} "
#             else:
#                 st.error("No conversation history available to convert!")
#         else:
#             try:
#                 chat_index = 2 * (int(chat_number) - 1)  # Convert to 0-based index
#                 if 0 <= chat_index < len(st.session_state.history):
#                     selected_chat = st.session_state.history[chat_index]
#                     full_text = f"Question: {selected_chat['message']} "
#                     if chat_index + 1 < len(st.session_state.history):
#                         assistant_response = st.session_state.history[chat_index + 1]
#                         full_text += f"Answer: {assistant_response['message']}"
                    
#                 else:
#                     st.error(f"Invalid chat number! Please enter a number between 1 and {len(st.session_state.history)}.")
#             except ValueError:
#                 st.error("Please enter a valid number or 'all' for full conversation.")

#         if full_text:
#             audio_file = text_to_speech_pyttsx3(full_text)
#             st.sidebar.audio(audio_file, format="audio/mp3")
#             st.sidebar.download_button("Download Audio", audio_file, file_name="chat_history_audio.mp3")

# # Main driver
# if __name__ == "__main__":
#     create_table()  # Ensure the table is created when the app starts
#     main()




# import streamlit as st
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from langchain.schema import HumanMessage
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import regex as re
# import os
# from dotenv import load_dotenv
# load_dotenv()

# groq_api_key = os.getenv('GROQ_API_KEY')
# def initialize_session_state():
#     """Initialize session state variables."""
#     if "quiz_data" not in st.session_state:
#         st.session_state.quiz_data = None
#     if "current_question_idx" not in st.session_state:
#         st.session_state.current_question_idx = 0
#     if "marks" not in st.session_state:
#         st.session_state.marks = 0
#     if "user_answers" not in st.session_state:
#         st.session_state.user_answers = {}
#     if "explanations" not in st.session_state:
#         st.session_state.explanations = {}

# def calculate_relevance(chat_text, questions):
#     """
#     Calculate relevance score using TF-IDF cosine similarity.
#     """
#     vectorizer = TfidfVectorizer()
#     all_texts = [chat_text] + [q[0] for q in questions]
#     tfidf_matrix = vectorizer.fit_transform(all_texts)
#     similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
#     return [round(score * 100, 2) for score in similarity_scores]  # Convert to percentage

# def generate_quiz_questions(difficulty, topic=None, num_questions=5, chat_history=None):
#     """
#     Generate a list of quiz questions based on the topic, difficulty, and number of questions.
#     """
#     # num_questions = max(5, min(num_questions, 15))
    
#     llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.7)

#     if not chat_history:
#         return None

#     chat_text = "\n".join(
#         [f"Human: {message.content}" if isinstance(message, HumanMessage) else f"AI: {message.content}" for message in chat_history]
#     )

#     if not topic:
#         topic_prompt = "Based on the following conversation, suggest a relevant topic for a quiz:"
#     else:
#         topic_prompt = f"Topic: {topic}"

#     prompt = PromptTemplate(
#         input_variables=["chat_text", "topic", "difficulty", "num_questions"],
#         template=f"""
#         {topic_prompt}
#         Conversation History:
#         {chat_text}

#         Based on this context, generate {num_questions} multiple-choice quiz questions with a difficulty level of '{difficulty}'.
#         Each question should:
#         - Be directly relevant to the topic and the content of the conversation.
#         - Include 4 answer options (A, B, C, D), with one clearly correct answer marked.
#         - Provide a clear and concise explanation after the correct answer.
#         - Don't provide any new lines after the options and the answer

#         Format your response as:
#         1. Question
#         A) Option 1
#         B) Option 2
#         C) Option 3
#         D) Option 4
#         Answer: <Correct Option>
#         Explanation: <Explanation for the answer>

#         Generate {num_questions} such questions in the specified format.
#         """
#     )

#     chain = prompt | llm | StrOutputParser()
#     question_data = chain.invoke({
#         "topic": topic,
#         "chat_history": chat_text,
#         "difficulty": difficulty,
#         "num_questions": num_questions
#     })
#     print(f"Generated prompt: {prompt}")
#     print("\n\n\n\n\n\n")
#     print(f"Generated question data: {question_data}")

#     return question_data.get("text", "") if hasattr(question_data, "get") else str(question_data),chat_text


# def parse_questions(question_data):
#     """
#     Parse multiple quiz questions from the generated data.
#     """
#     questions = []
#     question_blocks = question_data.strip().split("\n\n")  # Split questions based on double newlines

#     for block in question_blocks:
#         lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        
#         if len(lines) < 3:
#             continue

#         try:
#             question = lines[0]  # First line is the question
#             options = {}

#             # Extract options (A, B, C, D)
#             for line in lines[1:]:
#                 match = re.match(r"([A-D])\)\s*(.+)", line)  # Match option format like "A) Some text"
#                 if match:
#                     key, value = match.groups()
#                     options[key] = value.strip()
#                     if len(options) == 4:
#                         break

#             # Extract the correct answer, handling inconsistent spaces
#             answer_line = next(line for line in lines if line.startswith("Answer:"))
#             answer_match = re.search(r"Answer:\s*([A-D])\)\s*(.+)", answer_line)
            
#             if not answer_match:
#                 raise ValueError(f"Correct answer not found in question data:\n{block}")

#             correct_answer = answer_match.group(1)  # Extract letter (A, B, C, or D)
#             print("Correct answer is:", correct_answer)

#             # Extract explanation, if present
#             explanation = None
#             for line in lines:
#                 if line.startswith("Explanation:"):
#                     explanation = line.split("Explanation:", 1)[1].strip()
#                     break  # Stop after finding the first explanation

#             if not explanation:
#                 explanation = "No explanation provided."
#             questions.append((question, options, correct_answer, explanation))
#             print("Parsed question:", questions[-1])
            
#         except (IndexError, StopIteration):
#             raise ValueError("Error parsing question data: Correct answer missing.")

#     return questions



# def display_quiz_page(chat_history,topic=None):
#     """Main function to display the interactive quiz page."""
#     st.title("Interactive Quiz Application")
#     st.write("Generate and take a quiz on any topic!")

#     # Difficulty Selection (Radio)
#     difficulty = st.radio("📊 Select Difficulty Level:", ["Easy", "Medium", "Hard"], horizontal=True)

#     # Number of Questions (Slider)
#     num_questions_input = st.slider("🎯 Select Number of Questions:", min_value=3, max_value=30, value=5)


#     initialize_session_state()

#     num_questions = validate_num_questions(num_questions_input)

#     if st.button("Generate Quiz Question"):
#         if not topic:
#             st.write("No topic specified, AI is suggesting a relevant topic...")
#         with st.spinner("Generating questions..."):
#             try:
#                 question_data,chat_text = generate_quiz_questions(difficulty,topic, num_questions,chat_history)
#                 relevance_scores = calculate_relevance(chat_text, question_data)
#                 st.session_state.quiz_data = parse_questions(question_data)
#                 st.session_state.current_question_idx = 0
#                 st.session_state.relevance_scores = relevance_scores
#                 st.session_state.marks = 0
#                 st.session_state.user_answers = {}
#                 st.session_state.explanations = {}
#             except Exception as e:
#                 st.error(f"Failed to generate quiz questions: {str(e)}")

#     if st.session_state.quiz_data:
#         display_question()
#         # display_navigation_buttons()


# def validate_num_questions(input_value):
#     """Validate and return the number of questions."""
#     if input_value:
#         num = int(input_value)
#         return num
#     else:
#         st.warning("Invalid input for number of questions. Defaulting to 5.")
#         return 5


# def display_question():
#     """Display up to 5 quiz questions at a time with submit buttons and navigation."""
#     current_idx = st.session_state.current_question_idx
#     total_questions = len(st.session_state.quiz_data)
#     questions_per_page = 5
    
#     start_idx = (current_idx // questions_per_page) * questions_per_page
#     end_idx = min(start_idx + questions_per_page, total_questions)
    
#     for idx in range(start_idx, end_idx):
#         question, options, correct_answer, explanation = st.session_state.quiz_data[idx]
        
#         st.write(f"**Question {current_idx + 1}:** {question}")
#         for key, value in options.items():
#             st.write(f"**{key})** {value}")  # This ensures both key and value are displayed

#         user_answer_key = f"user_answer_{idx}"
#         submit_key = f"submit_{idx}"
        
#         user_answer = st.radio(
#             f"Select your answer for Question {idx + 1}:",
#             list(options.keys()),
#             key=user_answer_key,
#             disabled=user_answer_key in st.session_state.user_answers  # Disable if already submitted
#         )
        
#         if submit_key not in st.session_state.user_answers:
#             if st.button("Submit", key=submit_key):
#                 evaluate_answer(user_answer, correct_answer, explanation, idx)
#                 st.session_state.user_answers[submit_key] = True  # Mark as submitted
#                 # st.rerun()
#         else:
#             st.write("✅ Answer Submitted")
    
#     # Navigation buttons
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         if start_idx > 0:
#             if st.button("Previous Questions"):
#                 st.session_state.current_question_idx -= questions_per_page
#                 st.rerun()
    
#     with col2:
#         if end_idx < total_questions:
#             if st.button("Next Questions"):
#                 st.session_state.current_question_idx += questions_per_page
#                 st.rerun()
    
#     # Display final score when all questions are answered
#     if len(st.session_state.user_answers) == total_questions:
#         st.write(f"**Final Score:** {st.session_state.marks}/{total_questions}")
#         st.balloons()
    

# def evaluate_answer(user_answer, correct_answer, explanation,idx):
#     """Evaluate the user's answer and provide feedback."""
#     # Normalize the answers to uppercase or strip whitespace
#     user_answer = user_answer.strip()
#     correct_answer = correct_answer.strip()
#     print("user_answer is {0}".format(user_answer))
#     print("correct_answer is {0}".format(correct_answer[0]))

#     if user_answer == correct_answer[0]:
#         st.success("Correct! 🎉")
#         st.session_state.marks += 1
#     else:
#         st.error(f"Wrong. The correct answer is {correct_answer}.")
    
#     st.info(f"**Explanation:** {explanation}")
#     st.session_state.explanations[idx] = explanation










# def display_question():
#     """Display up to 5 quiz questions at a time with submit buttons and navigation."""
#     current_idx = st.session_state.current_question_idx
#     total_questions = len(st.session_state.quiz_data)
#     questions_per_page = 5

#     start_idx = (current_idx // questions_per_page) * questions_per_page
#     end_idx = min(start_idx + questions_per_page, total_questions)

#     for idx in range(start_idx, end_idx):
#         question, options, correct_answer, explanation = st.session_state.quiz_data[idx]

#         st.write(f"**Question {idx + 1}:** {question}")
#         for key, value in options.items():
#             st.write(f"**{key})** {value}")

#         user_answer_key = f"user_answer_{idx}"
#         submit_key = f"submit_{idx}"

#         # Check if user already answered
#         if idx in st.session_state.user_answers:
#             st.write(f"**✅ Your Answer:** {st.session_state.user_answers[idx]}")
#             st.write(f"**💡 Explanation:** {st.session_state.explanations[idx]}")
#         else:
#             # Allow user to select an answer
#             user_answer = st.radio(
#                 f"Select your answer for Question {idx + 1}:",
#                 list(options.keys()),
#                 key=user_answer_key
#             )

#             # Submit Button
#             if st.button(f"Submit", key=submit_key):
#                 evaluate_answer(user_answer, correct_answer, explanation, idx)
#                 st.session_state.user_answers[idx] = user_answer
#                 st.session_state.explanations[idx] = explanation
#                 st.rerun()  # Rerun to reflect changes instantly

#     # Navigation buttons
#     col1, col2 = st.columns([1, 1])

#     with col1:
#         if start_idx > 0:
#             if st.button("Previous Questions"):
#                 st.session_state.current_question_idx -= questions_per_page
#                 st.rerun()

#     with col2:
#         if end_idx < total_questions:
#             if st.button("Next Questions"):
#                 st.session_state.current_question_idx += questions_per_page
#                 st.rerun()

#     # Show final score after all questions are answered
#     if len(st.session_state.user_answers) == total_questions:
#         st.write(f"**Final Score:** {st.session_state.marks}/{total_questions}")
#         st.balloons()







# if __name__ == "__main__":
#     display_quiz_page()








