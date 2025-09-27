import atexit
import os
import pdfplumber
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from htmlTemplates import user_template, bot_template

api_key= os.environ['MY_ENV_VAR']
model_name = 'llama-3.3-70b-versatile'

groq_chat = ChatGroq(groq_api_key=api_key, model_name=model_name)

llm = groq_chat

## Do not mofify
def load_db(embeddings,pdf_path):
    text =''
    with open(pdf_path,'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()

    text_splitter=SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    docs = text_splitter.split_text(text)
    vectorstore = FAISS.from_texts(docs, embeddings)
    return vectorstore

embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')

pdf_path = "Serway.pdf"

#Do not modify this part
if not os.path.exists('faiss_index'):
    vectorstore=load_db(embeddings,pdf_path)
    vectorstore.save_local("faiss_index")
else:
    vectorstore = FAISS.load_local("faiss_index",embeddings=embeddings,allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever()

template = """Eres un asistente para contestar preguntas del usuario sobre ciruitos eléctricos. 
    Contesta siempre en español y agradece al usuario por preguntar. Si las preguntas son sobre otro tema, contesta que no puedes contestar.
    {context}
    Question: {question}
    Helpful Answer:"""
qa_prompt = ChatPromptTemplate.from_template(template)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, combine_docs_chain_kwargs={"prompt": qa_prompt})

def save_history():
        with open("chat_history.txt", "w", encoding="utf-8") as f:
            for q, a in history:
                f.write(f"Usuario: {q}\nAsistente: {a}\n\n")

history = []
st.header('My Chatbot')
st.write(bot_template.replace("{{MSG}}", "Hola, estoy aquí para ayudarte"), unsafe_allow_html=True)
question = st.chat_input("Pregúntame algo")
if question:
    st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
    result=conversation_chain({"question": question}, {"chat_history": history})
    st.write(bot_template.replace("{{MSG}}", result['answer']), unsafe_allow_html=True)
    history.append((question, result["answer"]))
    atexit.register(save_history)