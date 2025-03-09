from langchain.chains import LLMChain,StuffDocumentsChain,ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from utils.prompt_templates import memory_prompt_template
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
# import chromadb
import yaml
import os
from dotenv import load_dotenv

# Groq_API_key loading
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Loading config
with open("config.yaml") as f:
    config = yaml.safe_load(f)
    
# Method for creating LLM
def create_llm(model_path = config["model_path"]["llama"],temperature = config["model_path"]["temperature"]):    
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_path, temperature=temperature)
    return llm

# method for creating chat history
def create_chat_history(chat_history):
    return ConversationBufferWindowMemory(memory_key="history",chat_memory=chat_history, k=5)

#method for creating prompt
def create_prompt_template(template):
    return PromptTemplate.from_template(template)

# method for creating llm chain
def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm=llm, prompt=chat_prompt, memory=memory)

# method for loading normal chain
def load_normal_chain(chat_history):
    return chatChain(chat_history)


# class for llm chains
class chatChain:
    def __init__(self,chat_history):
        self.memory = create_chat_history(chat_history)
        llm = create_llm()
        chat_prompt = create_prompt_template(memory_prompt_template)
        self.llm_chain = LLMChain(llm = llm,prompt=chat_prompt,memory=self.memory)
        
    def run(self,user_input):
        return self.llm_chain.run(human_input=user_input,history = self.memory.chat_memory.messages ,stop=["Human:"])
        
