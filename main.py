import streamlit as st
import openai
import os
from dotenv import load_dotenv
import shutil

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_version = os.getenv('OPENAI_API_VERSION')
openai.api_base = os.getenv('OPENAI_API_BASE')

from src.utils import *

# Basic text for the app
st.title("Conversa con tus PDF's")
st.write('''Aplicación en etapa de testeo para que puedan interactuar con sus documentos por medio de un chat.
Actualmente se tiene la opción de crear nuevos modelos y cargar modelos creados con documentación ya subida.
Cualquier duda respecto al uso de la herramienta contactar a Patricio Ortiz : patricio@coddi.ai''')

# Define state variables
## History of messages
if "messages" not in st.session_state:
    st.session_state.messages = []

## Name of the experiment used in the session 
if "name_exp" not in st.session_state:
    st.session_state.name_exp = ""

## System role used in the session 
if "system_role" not in st.session_state:
    st.session_state.system_role = ""

## Validation of the experiment name
if "exp_bool" not in st.session_state:
    st.session_state.exp_bool = False

# Sidebar of the app
with st.sidebar:
    # Logo of the company
    st.image('img/logo.jpg')
    # Main option of actions for the user
    action = st.selectbox("¿Que deseas hacer?", ("","Cargar un modelo existente","Crear un nuevo modelo", ))

    # Case 1. Load a model
    if action == "Cargar un modelo existente":
        # Get experiments in memory
        store_list = os.listdir('Storage')
        # Select model
        exp_name = st.selectbox("Elige un experimento", [""] + store_list)
        # SIf model is selected
        if exp_name != "":
            # Save the name as state variable
            st.session_state.name_exp = exp_name
            # Change the validation of the experiment to True
            st.session_state.exp_bool = True
            # Obtain a description of the experiment
            st.session_state.system_role = get_personality(exp_name)
            get_docs(exp_name)

    # Case 2. Create a model
    elif action == "Crear un nuevo modelo":
        # Help text for user.
        st.header("Personalice su experimento")
        st.write('A continuación usted podrá personalizar el modelo de IA con el que interactuará, entregando dos paramétros:')
        st.write('*Personalidad del modelo : Rol que la IA tomará. Este es un parámtero opcional, con el cual pueden lograr resultados más personalizados.')
        st.write('Ejemplos de este campo son: "ingeniero mecánico con experiencia en equipos mineros" o "directivo de empresa buscando mejorar la productividad"')
        st.write('*Nombre del experimento : Con el cual podrá cargar su modelo en el futuro.')

        # user inputs to personalise the model.
        system_role = st.text_input("Personalidad del modelo", "")
        exp_name = st.text_input("Nombre del modelo", "")

        # Validate the name provided by the user
        exp_bool = check_exp_name(exp_name, st.session_state.name_exp)

        # If the name is valid
        if exp_bool:
            # Save the name of the experiment as state variable
            st.session_state.name_exp = exp_name

            # If a role is provided save it as .txt
            check_personality(system_role, st.session_state.name_exp)

            # Save the role of the agent as state variable
            st.session_state.system_role = system_role

            # Module to upload the documents
            st.header('Suba sus PDFs aquí')
            uploaded_files = st.file_uploader("Upload your documents", type="pdf", accept_multiple_files=True)
            
            # If the user uploads documents
            if uploaded_files:
                # Create a state value to validate the upload and avoid errors
                if 'qa' not in st.session_state:
                    # wheel to show the user that the documents are being processed
                    with st.spinner("Cargando documentos..."):
                        # Destroy the temp folder if it exists
                        # This could happen if a previous excecution of the app was interrupted
                        if 'temp' in os.listdir():
                            shutil.rmtree('temp')
                        # Create a temp repository to save the model
                        os.mkdir('temp')

                        # Save each uploaded file to the 'temp' folder
                        for uploaded_file in uploaded_files:
                            # Save each uploaded file to the 'temp' folder
                            with open(os.path.join('temp', uploaded_file.name), "wb") as f:
                                f.write(uploaded_file.getvalue())
                        # Create the index for the model -> the agent
                        vector_index = create_index('temp', st.session_state.name_exp)
                        # Save the name of the documents provided into a text file.
                        docs_uploaded(st.session_state.name_exp)
                        # Set the qa state variable to True in order to avoid errors
                        st.session_state['qa'] = True
                        # Once the model is created, set the state of the model to True
                        st.session_state.exp_bool = True
                        # Delete the temp folder
                        shutil.rmtree('temp')

# If a valid experiment is loaded
# This could be triggered in both use cases.
if st.session_state.exp_bool:

    # Display the previous messages from the session messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # If the user send the message
    if prompt := st.chat_input("What is up?"):

        # Display the message sent by the user
        with st.chat_message("user"):
            st.markdown(prompt)

        # with st.spinner('Pensando respuesta...')

        # Get the history of messages to sent to the model.
        chat_history = get_history(st.session_state.messages, st.session_state.system_role)
        
        # Create the agent using the history of messages
        agent = call_agent(chat_history=chat_history, exp_name=st.session_state.name_exp)

        # Save the message sent by the user
        st.session_state.messages.append({"role": "user", "content": prompt})

        st.session_state.answering = True

        if st.session_state.answering:

            with st.spinner('Pensando respuesta...'):
                # get the response from the model
                response = agent.chat(prompt)
                # get the response content from the model
                full_response = f"{response.response} \n {8*'_'}"
                mdata = response.metadata
                for ref in mdata.values():
                    full_response += f'\n\n doc: {ref["file_name"]} | page: {ref["page_label"]}'

            # add the response to the session messages
             # st.write('hola')
            # Display the response from the model
            with st.chat_message("assistant"):
                # set an empty space
                message_placeholder = st.empty()
                # display the response
                message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response, "metadata": response.metadata})
            st.session_state.answering = False
            
        # st.write('hola2')
        
        # Boton de reinicio de conversación.
        # if len(st.session_state.messages) >= 2:
        #     reset_button = st.button("Reiniciar conversación")
        #     st.write(reset_button)
        #     if reset_button:
        #         st.session_state.messages = []