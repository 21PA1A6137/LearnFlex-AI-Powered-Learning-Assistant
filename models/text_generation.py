from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import random
from models.image_generation import generate_image

def rewrite_query(user_query, llm):
    query_prompt = PromptTemplate(
        template="""
        Given the following student query, refine it to be more structured, clear
        while maintaining the original meaning.

        📝 **Original Query:** {query}  
        🔍 **Optimized Query:**  
        """,
        input_variables=["query"]
    )

    query_chain = LLMChain(llm=llm, prompt=query_prompt)
    return query_chain.run({"query": user_query})


def generate_explanation(topic, difficulty, llm, subject=None, learning_style="balanced"):
    """
    Generate a high-quality, dynamic explanation based on the topic, difficulty level, and subject.
    Supports learning styles, auto-generated code snippets, quiz questions, and image generation.
    """
    refined_query = rewrite_query(topic, llm)
    
    prompt = PromptTemplate(
        input_variables=["topic", "subject", "difficulty", "learning_style"],
        template="""
        You are an AI tutor providing structured, engaging, and insightful explanations.  
        Your task is to generate a **clear and relevant explanation** for **"{topic}"**  
        in the context of **"{subject}"**, adjusting to the **"{difficulty}"** level  
        and the learner's **"{learning_style}"** preference.  

        🔹 **Learning Style Customization:**  
        - **Visual Learner** 🎨: Use analogies, structured formatting, or simple diagrams **if truly needed**.  
        - **Text Learner** 📖: Provide a well-structured, logical explanation with real-world examples.  
        - **Hands-on Learner** 🛠️: Only provide relevant Python/Java code snippets **if coding is directly related**.  
        - **Balanced** ⚖️: Mix structured text, real-world applications, and **concise** interactive elements.  

        🔹 **Difficulty Levels:**  
        - **Easy:** Use simple language, analogies, and relatable examples.  
        - **Medium:** Provide structured insights with technical depth.  
        - **Hard:** Include deep technical explanations, research insights, and advanced applications.  

        ❗ **Important Guidelines:**  
        - ⚡ **Programming Topics?** If the topic is related to coding (e.g., Sorting Algorithms, Decision Trees),  
        provide a relevant Python/Java code snippet. Otherwise, exclude code.  
        - **Examples** 📝: If applicable, include a short example for better understanding.  
        - **Formula (if applicable)** ✏️: If the topic involves formulas, include them with a proper explanation.  
        - 🚫 **Do NOT generate ASCII diagrams** unless the topic involves a **visual concept** (e.g., trees, graphs).  
        - ❌ **Avoid unnecessary details, stories, or redundant information**. Keep it **exam-relevant**.  
        - ✅ Ensure the explanation is **concise, clear, and well-structured**. 
        - provide the code if the question is related to programming or related 

        ⚡ **Important Rules:**  
        1️ **General Knowledge / Factual Questions:**  
        - If the question is a **simple fact** (e.g., *"What is the capital of France?"*), **respond concisely** in 1-2 lines.  
        - **Do NOT** provide unnecessary structure, learning styles, or explanations.  

        2️ **Relevant Similar Exam Questions:**  
        - If the topic is **exam-relevant**, generate **3 related questions** that a student might encounter.  
        - **DO NOT** generate questions if they are not useful (e.g., for general/factual questions like what is the capital of India).  

        ✅ Now, generate the best possible explanation for **"{topic}"** at the **"{difficulty}"** level,  
        following the **"{learning_style}"** approach.
        """
    )


    chain = LLMChain(llm=llm, prompt=prompt)
    # chain = prompt | llm
    explanation = chain.run({"topic": refined_query, "difficulty": difficulty, "subject": subject, "learning_style": learning_style})
    
    extras = ""

    if learning_style.lower() == "hands-on":
        extras += generate_code_snippet(topic) + "\n\n"

    if learning_style.lower() == "visual":
        extras += generate_ascii_diagram(topic) + "\n\n"
        extras += generate_image(topic) + "\n\n"

    if learning_style.lower() == "balanced":
        extras += generate_code_snippet(topic) + "\n\n"
        extras += generate_ascii_diagram(topic) + "\n\n"
    return explanation + "\n\n" + extras



def generate_code_snippet(topic):
    """Generate a simple Python/Java code snippet based on the topic."""
    code_samples = {
        "Sorting Algorithms": """# Python - Bubble Sort\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\nprint(bubble_sort([64, 34, 25, 12, 22, 11, 90]))""",
        "Decision Trees": """# Python - Basic Decision Tree using scikit-learn\nfrom sklearn.tree import DecisionTreeClassifier\nX = [[0, 0], [1, 1]]  # Sample data\ny = [0, 1]  # Labels\nclf = DecisionTreeClassifier().fit(X, y)\nprint(clf.predict([[2, 2]]))"""
    }
    return code_samples.get(topic, "")


def generate_ascii_diagram(topic):
    """Generate a simple ASCII diagram representation for a topic."""
    diagrams = {
        "Decision Tree": """
        Root
        ├── Left Child
        │   ├── Leaf (Yes)
        └── Right Child
            ├── Leaf (No)
        """,
        "Neural Network": """
        [Input Layer] → [Hidden Layer] → [Output Layer]
        """
    }
    return diagrams.get(topic, "")


