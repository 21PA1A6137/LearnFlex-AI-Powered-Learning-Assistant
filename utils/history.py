import json
from datetime import datetime
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain


def save_chat_history_json(chat_history, file_path):
    """Save chat history in JSON format."""
    with open(file_path, "w") as f:
        json_data = [message.dict() for message in chat_history]
        json.dump(json_data, f)

def generate_summary(chat_history, llm):
    """Generate a descriptive summary for the chat history."""
    # Combine user messages for the summary
    conversation = " ".join([msg.content for msg in chat_history if msg.type == "human"])
    
    # Generate a summary from the LLM
    prompt = (
        "Generate a short (5-7 words) title-like summary for this conversation, "
        "suitable for naming a file. Keep it meaningful and concise: "
        f"{conversation}"
    )
    # prompt = f"Summarize this chat briefly  in a way suitable for naming a file: {conversation}"
    summary = llm.predict(prompt).strip()
    
    # Ensure the filename is safe for the filesystem
    sanitized_summary = "".join(c if c.isalnum() or c in " _-" else "_" for c in summary).strip()
    return sanitized_summary

def load_chat_history_json(file_path):
    """Load chat history from a JSON file."""
    with open(file_path, "r") as f:
        json_data = json.load(f)
        messages = [
            HumanMessage(**message) if message["type"] == "human" else AIMessage(**message)
            for message in json_data
        ]
        return messages

def get_time_stamp():
    """Get the current timestamp."""
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")



    
    
