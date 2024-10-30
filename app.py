import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import WebBaseLoader
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()


g_key = os.getenv("g_key")

model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=g_key,
                             temperature=0.2,convert_system_message_to_human=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=g_key)

store_path = 'notebooks/chroma_langchain_db'
if os.path.exists(store_path):
    print("we founf it")
vector_db = Chroma(collection_name="vbc_billing",embedding_function=embeddings, persist_directory=store_path, )

memory = ConversationBufferMemory(
    memory_key="chat_history",return_messages=True
)


qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_db.as_retriever(),
    memory=memory
)

print(memory.)

st.title("VBC Support Chat bot")

# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []

# user_input = st.chat_input("Ask Your Question ")

# if user_input:
#     conversation = qa_chain({"query": user_input})
#     bot_response = conversation['result']

#     # Store conversation
#     st.session_state["chat_history"].append(("You", user_input))
#     st.session_state["chat_history"].append(("Bot", bot_response))

# # Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask Your Question "):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    conversation = qa_chain({"query": prompt})
    bot_response = conversation['result']
    response = bot_response
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


