import streamlit as st
import numpy as np
from langchain_community.llms import Replicate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever

import os
st.title("Chat with stocks of Apple and Meta")
st.subheader('Since 2022')

if "messages" not in st.session_state:
    st.session_state.messages= []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message["content"])

# Display user input in the sidebar
st.sidebar.title("Replicate_API")
st.sidebar.text("Login to Replicate and copy Api token of replicate")
user_api = st.sidebar.text_input("Enter Text")

os.environ['REPLICATE_API_TOKEN'] = user_api 
from pypdf import PdfReader

directory = "stocks"

def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

documents = load_docs(directory)

    
def split_docs(documents,chunk_size=1000,chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)



def load_llm():
    llm = Replicate(
    model= "lucataco/phi-2:740618b0c24c0ea4ce5f49fcfef02fcd0bdd6a9f1b0c5e7c02ad78e9b3b190a6",
    input={"temperature": 0.75, "max_length": 2000}
    )
    return llm

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings)
persist_directory = "chroma_db"

vectordb = Chroma.from_documents(
documents=docs, embedding=embeddings, persist_directory=persist_directory)

vectordb.persist()
llm = load_llm()
new_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
#prompt::

if prompt := st.chat_input("Know about stocks"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        prom = prompt

    import random
    import time
    # Display assistant response in chat message container

    def response_generator(query):
        matching_docs = db.similarity_search(query)
        chain = load_qa_chain(llm=llm, chain_type="stuff",verbose=True)
        response =  chain.run(input_documents=matching_docs, question=query)
        

        for word in response.split():
            yield word + " "
            time.sleep(0.05) #retrieval_chain.run(query)

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prom))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
