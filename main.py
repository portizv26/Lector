import streamlit as st

import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

from src.utils import *

st.title("Chat with your PDF's")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    chat_history = get_history(st.session_state.messages)
    # st.write(st.session_state.messages)
    agent = call_agent(chat_history=chat_history)

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

