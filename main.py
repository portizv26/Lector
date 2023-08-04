import streamlit as st

import openai
import os
from dotenv import load_dotenv
import shutil

load_dotenv()

i=0

if i==1:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")

from src.utils import *
# st.write(os.listdir())
st.title("Chat with your PDF's")

with st.sidebar:
    st.header("Nombre su experimento")
    exp_name = st.text_input("Nombre del experimento", "")
    st.header('Suba sus PDFs aquí')
    uploaded_files = st.file_uploader("Upload your documents", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        if 'qa' not in st.session_state:
            with st.spinner("Loading documents..."):
                # shutil.rmtree('temp')
                
                os.mkdir('temp')
                for uploaded_file in uploaded_files:
                    # Save each uploaded file to the 'temp' folder
                    with open(os.path.join('temp', uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getvalue())

                vector_index = create_index('temp', exp_name)

                st.session_state['qa'] = True
                shutil.rmtree('temp')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    chat_history = get_history(st.session_state.messages)
    # st.write(st.session_state.messages)
    agent = call_agent(chat_history=chat_history, exp_name=exp_name)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response = agent.chat(prompt)
        # st.write(response)
        full_response += response.response
        full_response += f"\n {8*'_'}"
        mdata = response.metadata
        for ref in mdata.values():
            string_mdata = f'\n\n\n doc: {ref["file_name"]} | page: {ref["page_label"]}'
            full_response += string_mdata
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response, "metadata": response.metadata})

