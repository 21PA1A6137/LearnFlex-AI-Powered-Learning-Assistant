from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

def generate_quiz_question(topic, difficulty, llm, subject):
    """
    Generate a quiz question based on the topic, difficulty level, and subject.
    """
    # Define prompts for each difficulty level
    if difficulty.lower() == "easy":
        prompt = PromptTemplate(
            input_variables=["topic", "subject"],
            template="""
                Create a simple multiple-choice question on the topic '{topic}' from the subject '{subject}'.
                The question should be straightforward and include 4 answer options, where only one option is correct.
                Clearly mark the correct answer and ensure it's suitable for beginners or school-level students.
            """
        )
    elif difficulty.lower() == "medium":
        prompt = PromptTemplate(
            input_variables=["topic", "subject"],
            template="""
                Generate a thought-provoking quiz question on the topic '{topic}' from the subject '{subject}'.
                The question should challenge a college-level student, include 4 answer options, and mark the correct answer.
                Include explanations for why the correct answer is right, but keep the question concise and relevant.
            """
        )
    elif difficulty.lower() == "hard":
        prompt = PromptTemplate(
            input_variables=["topic", "subject"],
            template="""
                Develop an advanced quiz question on the topic '{topic}' from the subject '{subject}' for a professional audience.
                The question should be in-depth, requiring advanced knowledge of the subject. Include 4 detailed answer options, mark the correct answer, and provide a brief explanation of the reasoning behind the correct choice.
            """
        )
    else:
        raise ValueError("Invalid difficulty level. Please choose 'easy', 'medium', or 'hard'.")

    # Create the LLM chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain to generate a quiz question
    question_data = chain.run({"topic": topic, "subject": subject})

    return question_data
