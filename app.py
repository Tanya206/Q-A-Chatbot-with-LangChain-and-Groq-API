import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking 
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With OLLAMA"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,llm, temperature, max_tokens):
    llm = ChatGroq(model = llm)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer 


## Title of the App
st.title("Q&A Chatbot")

## Sidebar for Settings
# st.sidebar.title("Settings")
#api_key = st.sidebar.text_input("Enter your LLM API Key: ", type="password")

## Drop down to select various models
llm = st.sidebar.selectbox("Select Open Source Model", ["Llama3-8b-8192","mixtral-8x7b-32768","Gemma-7b-it"])

## Adjust Response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value = 50, max_value= 300, value=150)

## Main Interface for User Input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide a query.")
