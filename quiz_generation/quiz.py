import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import regex as re
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
def initialize_session_state():
    """Initialize session state variables."""
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None
    if "current_question_idx" not in st.session_state:
        st.session_state.current_question_idx = 0
    if "marks" not in st.session_state:
        st.session_state.marks = 0
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}
    if "relevance_scores" not in st.session_state:
        st.session_state.relevance_scores = []
    if "explanations" not in st.session_state:
        st.session_state.explanations = {}
    if "submitted_answers" not in st.session_state:
        st.session_state.submitted_answers = []

    # if "quiz_generated" not in st.session_state:
    #     st.session_state.quiz_generated = False
    # if "quiz_step" not in st.session_state:  # Track the current step
    #     st.session_state.quiz_step = "options"


def calculate_relevance(chat_text, questions):
    """
    Calculate relevance score using TF-IDF cosine similarity.
    """
    vectorizer = TfidfVectorizer()
    all_texts = [chat_text] + [q[0] for q in questions]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return [round(score * 100, 2) for score in similarity_scores]  # Convert to percentage

def generate_quiz_questions(difficulty, topic=None, num_questions=5, chat_history=None):
    """
    Generate a list of quiz questions based on the topic, difficulty, and number of questions.
    """
    # num_questions = max(5, min(num_questions, 15))
    
    llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.7)

    if not chat_history:
        return None

    chat_text = "\n".join(
        [f"Human: {message.content}" if isinstance(message, HumanMessage) else f"AI: {message.content}" for message in chat_history]
    )

    if not topic:
        topic_prompt = "Based on the following conversation, suggest a relevant topic for a quiz:"
    else:
        topic_prompt = f"Topic: {topic}"

    prompt = PromptTemplate(
        input_variables=["chat_text", "topic", "difficulty", "num_questions"],
        template="""
        {topic}

        📌 **Conversation History:**  
        {chat_text}

        ✍️ **Task:**  
        Generate **{num_questions}** multiple-choice quiz questions at the **{difficulty}** level.  
        
        ✅ **Each question must:**  
        - Be directly related to the topic and conversation.  
        - Have **4 answer choices** (A, B, C, D).  
        - Clearly indicate the correct answer.  
        - Provide a **concise explanation** for the answer.  
        - **Do NOT insert blank lines between the options and the answer.**  

        📖 **Format:**  
        ```
        1. Question  
        A) Option 1  
        B) Option 2  
        C) Option 3  
        D) Option 4  
        Answer: <Correct Option>  
        Explanation: <Brief explanation>  
        ```  
        
        Generate **{num_questions}** questions in the above format.
        """
    )

    chain = prompt | llm | StrOutputParser()
    question_data = chain.invoke({
        "topic": topic,
        "chat_text": chat_text,
        "difficulty": difficulty,
        "num_questions": num_questions
    })
    print(f"Generated prompt: {prompt}")
    print("\n\n\n\n\n\n")
    print(f"Generated question data: {question_data}")

    return question_data.get("text", "") if hasattr(question_data, "get") else str(question_data),chat_text


def parse_questions(question_data):
    """
    Parse multiple quiz questions from the generated data.
    """
    questions = []
    question_blocks = question_data.strip().split("\n\n")  # Split questions based on double newlines

    for block in question_blocks:
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        
        if len(lines) < 3:
            continue

        try:
            question = lines[0]  # First line is the question
            options = {}

            # Extract options (A, B, C, D)
            for line in lines[1:]:
                match = re.match(r"([A-D])\)\s*(.+)", line)  # Match option format like "A) Some text"
                if match:
                    key, value = match.groups()
                    options[key] = value.strip()
                    if len(options) == 4:
                        break

            # Extract the correct answer, handling inconsistent spaces
            answer_line = next(line for line in lines if line.startswith("Answer:"))
            answer_match = re.search(r"Answer:\s*([A-D])\)?", answer_line)
            
            if not answer_match:
                raise ValueError(f"Correct answer not found in question data:\n{block}")

            correct_answer = answer_match.group(1)  # Extract letter (A, B, C, or D)
            print("Correct answer is:", correct_answer)

            # Extract explanation, if present
            explanation = None
            for line in lines:
                if line.startswith("Explanation:"):
                    explanation = line.split("Explanation:", 1)[1].strip()
                    break  # Stop after finding the first explanation

            if not explanation:
                explanation = "No explanation provided."
            questions.append((question, options, correct_answer, explanation))
            print("Parsed question:", questions[-1])
            
        except (IndexError, StopIteration):
            raise ValueError("Error parsing question data: Correct answer missing.")

    return questions



def display_quiz_page(chat_history,topic=None):
    """Main function to display the interactive quiz page."""
    st.title("Interactive Quiz Application")
    st.write("Generate and take a quiz on any topic!")

    # Difficulty Selection (Radio)
    difficulty = st.radio("📊 Select Difficulty Level:", ["Easy", "Medium", "Hard"], horizontal=True,index=1)

    # Number of Questions (Slider)
    num_questions_input = st.slider("🎯 Select Number of Questions:", min_value=3, max_value=30, value=5)


    initialize_session_state()

    num_questions = validate_num_questions(num_questions_input)

    if st.button("Generate Quiz Question"):
        if not topic:
            st.write("No topic specified, AI is suggesting a relevant topic...")
        with st.spinner("Generating questions..."):
            try:
                question_data,chat_text = generate_quiz_questions(difficulty,topic, num_questions,chat_history)
                relevance_scores = calculate_relevance(chat_text, question_data)
                st.session_state.quiz_data = parse_questions(question_data)
                st.session_state.current_question_idx = 0
                st.session_state.relevance_scores = relevance_scores
                st.session_state.marks = 0
                st.session_state.user_answers = {}
                st.session_state.explanations = {}
                
            except Exception as e:
                st.error(f"Failed to generate quiz questions: {str(e)}")

    if st.session_state.quiz_data:
        display_question()

    def generate_quiz_report():
        """Generate a downloadable quiz report."""
        report = "📋 Quiz Report\n\n"
        for i, (question, options, correct_answer, explanation) in enumerate(st.session_state.quiz_data):
            user_answer = st.session_state.user_answers[i]
            report += f"Q{i+1}: {question}\n"
            for key, value in options.items():
                report += f"  {key}) {value}\n"
            report += f"✅ Correct Answer: {correct_answer}\n"
            report += f"📝 Your Answer: {user_answer}\n"
            report += f"💡 Explanation: {explanation}\n\n"
        total_questions = len(st.session_state.quiz_data)
        report += f"\n**Total Marks:** {st.session_state.marks}/{total_questions}\n"
        return report

    if st.button("📥 Download Quiz Report"):
        report_text = generate_quiz_report()
        st.download_button(label="Download", data=report_text, file_name="quiz_report.txt", mime="text/plain")

def validate_num_questions(input_value):
    """Validate and return the number of questions."""
    if input_value:
        num = int(input_value)
        return num
    else:
        st.warning("Invalid input for number of questions. Defaulting to 5.")
        return 5


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
#             st.write(f"**{key})** {value}")  # This ensures both key and value are displayed

#         user_answer_key = f"user_answer_{idx}"
#         submit_key = f"submit_{idx}"
        
#         if idx in st.session_state.user_answers:
#             user_answer = st.session_state.user_answers[idx]
#             if user_answer == correct_answer[0]:
#                 st.success(f"✅ Correct! Your Answer: {user_answer}")
#             else:
#                 st.error(f"❌ Wrong! Your Answer: {user_answer}. The correct answer is {correct_answer}.")

#             st.info(f"💡 Explanation: {st.session_state.explanations[idx]}")
#         else:
#             user_answer = st.radio(
#                 f"Select your answer for Question {idx + 1}:",
#                 list(options.keys()),
#                 key=user_answer_key,
#                 disabled=user_answer_key in st.session_state.user_answers  # Disable if already submitted
#             )
        
#         if idx not in st.session_state.user_answers:
#             if st.button("Submit", key=submit_key):
#                 evaluate_answer(user_answer, correct_answer, explanation, idx)
#                 st.session_state.user_answers[submit_key] = True
#                 st.session_state.user_answers[idx] = user_answer
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

def display_question():
    """Display up to 5 quiz questions at a time with submit buttons and navigation."""
    current_idx = st.session_state.current_question_idx
    total_questions = len(st.session_state.quiz_data)
    questions_per_page = 5

    start_idx = (current_idx // questions_per_page) * questions_per_page
    end_idx = min(start_idx + questions_per_page, total_questions)

    for idx in range(start_idx, end_idx):
        question, options, correct_answer, explanation = st.session_state.quiz_data[idx]

        st.write(f"**Question {idx + 1}:** {question}")
        for key, value in options.items():
            st.write(f"**{key})** {value}")  # Ensure both key and value are displayed

        user_answer_key = f"user_answer_{idx}"
        submit_key = f"submit_{idx}"

        if idx in st.session_state.submitted_answers:  # Check if this question was already submitted
            user_answer = st.session_state.user_answers[idx]
            if user_answer == correct_answer[0]:
                st.success(f"✅ Correct! Your Answer: {user_answer}")
            else:
                st.error(f"❌ Wrong! Your Answer: {user_answer}. The correct answer is {correct_answer}.")

            st.info(f"💡 Explanation: {st.session_state.explanations[idx]}")
        else:
            user_answer = st.radio(
                f"Select your answer for Question {idx + 1}:",
                list(options.keys()),
                key=user_answer_key,
            )

            if st.button("Submit", key=submit_key):
                evaluate_answer(user_answer, correct_answer, explanation, idx)
                st.session_state.user_answers[idx] = user_answer
                st.session_state.submitted_answers.append(idx)  # Track submission separately
                st.rerun()

    # Navigation buttons
    col1, col2 = st.columns([1, 1])

    with col1:
        if start_idx > 0:
            if st.button("Previous Questions"):
                st.session_state.current_question_idx -= questions_per_page
                st.rerun()

    with col2:
        if end_idx < total_questions:
            if st.button("Next Questions"):
                st.session_state.current_question_idx += questions_per_page
                st.rerun()

    # Display final score when ALL questions are submitted
    if len(st.session_state.submitted_answers) == total_questions:
        st.write(f"**Final Score:** {st.session_state.marks}/{total_questions} 🎯")
        # st.balloons()

    

def evaluate_answer(user_answer, correct_answer, explanation,idx):
    """Evaluate the user's answer and provide feedback."""
    # Normalize the answers to uppercase or strip whitespace
    user_answer = user_answer.strip()
    correct_answer = correct_answer.strip()
    # print("user_answer is {0}".format(user_answer))
    # print("correct_answer is {0}".format(correct_answer[0]))
    st.session_state.user_answers[idx] = user_answer
    st.session_state.explanations[idx] = explanation
    if user_answer == correct_answer[0]:
        st.success("Correct! 🎉")
        st.session_state.marks += 1
    else:
        st.error(f"❌ Wrong! Your Answer: {user_answer}. The correct answer is {correct_answer[0]}.")
    st.info(f"💡 Explanation: {explanation}")


if __name__ == "__main__":
    display_quiz_page()

