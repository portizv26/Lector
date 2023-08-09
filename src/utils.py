# General libraries for vector storing
import openai
from llama_index import (
    SimpleDirectoryReader,
    GPTListIndex,
    GPTVectorStoreIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    set_global_service_context
    )

# language model to use
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
# Embeding model to use
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding

from llama_index.chat_engine.condense_question import CondenseQuestionChatEngine

# History messages
from llama_index.prompts  import Prompt
from llama_index.llms.base import ChatMessage, MessageRole

import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_version = os.getenv('OPENAI_API_VERSION')
openai.api_base = os.getenv('OPENAI_API_BASE')

def create_index(path, exp_name):
    # Define constants.
    max_input = 4096
    tokens = 500
    chunk_size_limit = 1000 
    chunk_overlap_ratio = 0.2

    # Initializing the PromptHelper with defined parameters.
    promptHelper = PromptHelper(context_window=max_input,
                                num_output=tokens,
                                chunk_overlap_ratio=chunk_overlap_ratio,
                                chunk_size_limit=chunk_size_limit
                                )
    # Initializing the LLMPredictor using the OPENAI model.
    llmPredictor = LLMPredictor(llm=AzureChatOpenAI(deployment_name='gpt-35-turbo',
                                                    temperature=0,
                                                    ),
                                )
    # llmPredictor = LLMPredictor(llm=ChatOpenAI(temperature=0,
    #                                            ),
    #                             )
    
    # Initializing the embedding model.
    embed_model = LangchainEmbedding(OpenAIEmbeddings())

    # Loading data from the provided path (this assumes it's a directory containing .pdf files).
    docs = SimpleDirectoryReader(path).load_data()

    # Creating a vector index using the loaded documents and defined models.
    service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor,
                                                   prompt_helper=promptHelper,
                                                   embed_model=embed_model,
                                                   chunk_size=chunk_size_limit,
                                                   chunk_overlap=chunk_size_limit*chunk_overlap_ratio,
                                                   context_window=max_input,
                                                   num_output=tokens,
                                                   )
    # Setting the global service context.
    set_global_service_context(service_context)

    # Creating the GPTVectorStoreIndex from the loaded documents.
    vectorIndex = GPTVectorStoreIndex.from_documents(documents=docs,
                                                     service_context=service_context,
                                                     show_progress=False,
                                                     )
    # Persisting the created vector index to storage.
    vectorIndex.storage_context.persist(persist_dir=f'Storage/{exp_name}')

    return vectorIndex

def load_index(exp_name=''):
    # Initialize a storage context based on the experiment name provided.
    storage_context = StorageContext.from_defaults(persist_dir = f'Storage/{exp_name}')
    
    # Load the index from the storage using the initialized context.
    index = load_index_from_storage(storage_context)

    return index
    

def call_agent(chat_history, exp_name):
    # Check if experiment name is empty.
    # This could happen if the user is trying to call an agent that has not been created yet.
    if exp_name == '':
        return None
    else:
        # Load the stored index based on the experiment name.
        index = load_index(exp_name=exp_name)
        
        # Get the query engine for the loaded index.
        query_engine = index.as_query_engine()

        # Initialize the chat engine with the query engine and chat history provided.
        chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine, 
            chat_history=chat_history,
            verbose=False
        )
        return chat_engine
    
def get_role(system_role):
    # If no role has been assigned set a deafult system role.
    if system_role == "":
        chat_history=[
                ChatMessage(role="system", content="Eres un asistente muy útil. Responde en español.")
                ]
    else:
        # Use the provided system role for the message.
        chat_history=[
                ChatMessage(role="system", content=f"Eres un {system_role.lower()}. Responde en español.")
                ]
    return chat_history

def get_history(session_messages, system_role):
    # Initialize chat history with the system role.
    chat_history = get_role(system_role)

    # If there's previous session messages, append it to the chat history.
    if len(session_messages) > 0:
        for message in session_messages:
            # If the message is from a user append the role and the content
            if message['role'] == 'user':
                role = MessageRole.USER
                chat_history.append(ChatMessage(role=role, 
                                                content=message['content']))
            # If the message is from an assistant append the role, content and metadata
            elif message['role'] == 'assistant':
                role = MessageRole.ASSISTANT
                chat_history.append(ChatMessage(role=role, 
                                                content=message['content'], 
                                                metadata=message['metadata']))
    
    return chat_history
    
def check_exp_name(exp_name, session_exp_name):
    '''
    exp_name: Experiment name provided by user.
    session_string: Experiment name of the streamlit session.
    '''
    
    # Check value of the session_exp_name
    # If the value is not empty, means that a agent is already assigned.
    if session_exp_name != "":
        return True
    # If no agent is assigned, check if of the exp_name provided by user is valid.
    else:
        # If experiment name already exists in storage.
        if exp_name in os.listdir('Storage'):
            st.error('El nombre de experimento ya existe, por favor elija otro.')
            return False
        # If experiment name is empty.
        elif (exp_name == ''):
            st.error('Ingrese un nombre de experimento.')
            return False
        else:
            return True
        
def get_docs(exp_name):       

    # If docs file is on directory, write the documents on screen
    if 'docs.txt' in os.listdir(f'Storage/{exp_name}'):
        p = f'Storage/{exp_name}/docs.txt'
        with open(p) as f:  
            documents_uploaded = f.read()
        st.write(f'{8*"____"}\n{documents_uploaded}')

def get_personality(exp_name):      
    # If role file is on directory, write the personality on screen and return it
    if 'role.txt' in os.listdir(f'Storage/{exp_name}'):
        p = f'Storage/{exp_name}/role.txt'
        with open(p) as f:  
            contents = f.read()
        personality = contents.title()
        st.write(f'{8*"____"}\nLa personalidad del modelo es: {personality}')
        return personality
    else:
        return ""
        
def check_personality(system_role, exp_name):
    # If a role is provided, save it to storage.
    if system_role != "":
        # Create a directory for the experiment.
        try:
            os.makedirs(f'Storage/{exp_name}')
        except:
            pass
        # Save the role to a text file.
        f = open(f"Storage/{exp_name}/role.txt", "w")
        f.write(system_role)
        f.close()

def docs_uploaded(exp_name):
    # Make sure a directory exists for the model
    ## This could be avoidable, but I have it just in case.
    try:
        os.makedirs(f'Storage/{exp_name}')
    except:
        pass
    # Get the names of the documents uploaded, without the extension.
    docs = [i.removesuffix('.pdf') for i in os.listdir('temp')]
    # Save the names into an easy string
    string = 'El modelo fue entrenado con los siguientes PDFs:'
    for doc in docs:
        string += f'\n\n{doc}'
    # Save the string to a text file. 
    f = open(f"Storage/{exp_name}/docs.txt", "w")
    f.write(string)
    f.close()

#TODO LIST:
# 1. Obtener la metadata de las respuestas -> done
# 2. Que funcione con credenciales de azure -> done -> problemas
# 3. Desplegar online -> done
# 4. Modulo para subir resultados. -> done
# 5. personalizar system. -> done
# 6. Elegir modelo anterior -> done
# 6,5 -> Explicar que tiene modelo anterior. -> done
# 7. logo -> done
# 8. lógica para nombre de experimento -> done

# # list of `ChatMessage` objects
# custom_chat_history = [
#     ChatMessage(
#         role=MessageRole.USER, 
#         content='Hello assistant, we are having a insightful discussion about Paul Graham today.'
#     ), 
#     ChatMessage(
#         role=MessageRole.ASSISTANT, 
#         content='Okay, sounds good.'
#     )
# ]