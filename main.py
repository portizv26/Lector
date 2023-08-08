import streamlit as st
import openai
import os
from dotenv import load_dotenv
import shutil

openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_type = os.getenv('OPENAI_API_TYPE')
# openai.api_version = os.getenv('OPENAI_API_VERSION')
# openai.api_base = os.getenv('OPENAI_API_BASE')

from src.utils import *

st.title("Conversa con tus PDF's")
st.write('''Aplicación en etapa de testeo para que puedan interactuar con sus documentos por medio de un chat.
Actualmente se tiene la opción de crear nuevos modelos y cargar modelos creados con documentación ya subida.
Cualquier duda respecto al uso de la herramienta contactar a Patricio Ortiz : patricio@coddi.ai''')

if "messages" not in st.session_state:
    st.session_state.messages = []

if "name_exp" not in st.session_state:
    st.session_state.name_exp = ""

if "exp_bool" not in st.session_state:
    st.session_state.exp_bool = False

with st.sidebar:
    st.image('img/logo.jpg')
    action = st.selectbox("¿Que deseas hacer?", ("","Cargar un modelo existente","Crear un nuevo modelo", ))

    if action == "Cargar un modelo existente":
        store_list = os.listdir('Storage')
        # st.write(store_list)
        exp_name = st.selectbox("Elige un experimento", [""] + store_list)
        if exp_name != "":
            st.session_state.name_exp = exp_name
            st.session_state.exp_bool = True
            get_personality(exp_name)
            get_docs_uploaded(exp_name)

    elif action == "Crear un nuevo modelo":
        st.header("Personalice su experimento")
        st.write('A continuación usted podrá personalizar el modelo de IA con el que interactuará, entregando dos paramétros:')
        st.write('*Personalidad del modelo : Rol que la IA tomará. Este es un parámtero opcional, con el cual pueden lograr resultados más personalizados.')
        st.write('Ejemplos de este campo son: "ingeniero mecánico con experiencia en equipos mineros" o "directivo de empresa buscando mejorar la productividad"')
        st.write('*Nombre del experimento : Con el cual podrá cargar su modelo en el futuro.')
        system_role = st.text_input("Personalidad del modelo", "")
        exp_name = st.text_input("Nombre del modelo", "")

        exp_bool = check_exp_name(exp_name, st.session_state.name_exp)
        if exp_bool:            
            system_bool = check_role(system_role, exp_name)
            st.session_state.name_exp = exp_name
            st.header('Suba sus PDFs aquí')
            uploaded_files = st.file_uploader("Upload your documents", type="pdf", accept_multiple_files=True)
            if uploaded_files:
                if 'qa' not in st.session_state:
                    with st.spinner("Cargando documentos..."):
                        if 'temp' in os.listdir():
                            shutil.rmtree('temp')
                        # shutil.rmtree('temp')
                        
                        os.mkdir('temp')
                        for uploaded_file in uploaded_files:
                            # Save each uploaded file to the 'temp' folder
                            with open(os.path.join('temp', uploaded_file.name), "wb") as f:
                                f.write(uploaded_file.getvalue())

                        docs_uploaded(exp_name)

                        vector_index = create_index('temp', exp_name)

                        st.session_state['qa'] = True
                        st.session_state.exp_bool = True
                        shutil.rmtree('temp')
                    

if st.session_state.exp_bool:

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        chat_history = get_history(st.session_state.messages, system_role=system_role)
        # st.write(st.session_state.messages)
        agent = call_agent(chat_history=chat_history, exp_name=st.session_state.name_exp)

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

