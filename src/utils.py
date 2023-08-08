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
# openai.api_key = st.secrets["OPENAI_API_KEY"]
# openai.api_type = st.secrets['OPENAI_API_TYPE']
# openai.api_version = st.secrets['OPENAI_API_VERSION']
# openai.api_base = st.secrets['OPENAI_API_BASE']

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_version = os.getenv('OPENAI_API_VERSION')
openai.api_base = os.getenv('OPENAI_API_BASE')

# openai.api_key = os.getenv("API_KEY")
# st.write(openai.api_key)

def create_index(path, exp_name=''):
    max_input = 4096
    tokens = 500
    chunk_size_limit = 1000 #for LLM, we need to define chunk size
    chunk_overlap_ratio = 0.2

    #define prompt
    promptHelper = PromptHelper(context_window = max_input,
                                num_output = tokens,
                                chunk_overlap_ratio = chunk_overlap_ratio,
                                chunk_size_limit = chunk_size_limit
                                )
    #define LLM — there could be many models we can use, but in this example, let’s go with OpenAI model
    llmPredictor = LLMPredictor(llm=AzureChatOpenAI(deployment_name='gpt-35-turbo',
                                                    temperature=0,
                                                    ),
                                )
    # llmPredictor = LLMPredictor(llm=ChatOpenAI(temperature=0,
    #                                                 ),
    #                             )
    embed_model = LangchainEmbedding(OpenAIEmbeddings())

    #load data — it will take all the .pdf files, if there are more than 1
    docs = SimpleDirectoryReader(path).load_data()

    #create vector index
    service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor,
                                                   prompt_helper=promptHelper,
                                                   embed_model=embed_model,
                                                   chunk_size=chunk_size_limit,
                                                   chunk_overlap=chunk_size_limit*chunk_overlap_ratio,
                                                   context_window = max_input,
                                                   num_output = tokens,
                                                   )
    set_global_service_context(service_context)

    vectorIndex = GPTVectorStoreIndex.from_documents(documents=docs,
                                                     service_context=service_context,
                                                     show_progress=False,
                                                     )
    
    vectorIndex.storage_context.persist(persist_dir = f'Storage/{exp_name}')

    return vectorIndex

def load_index(exp_name=''):
    storage_context = StorageContext.from_defaults(persist_dir = f'Storage/{exp_name}')
    index = load_index_from_storage(storage_context)

    return index

def call_agent(chat_history, exp_name=''):
    if exp_name == '':
        return None
    else:
        index=load_index(exp_name=exp_name)
        query_engine = index.as_query_engine()

        chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine, 
            # condense_question_prompt=custom_prompt,
            chat_history=chat_history,
            verbose=False
        )
        return chat_engine

def get_history(session_list, system_role=""):
    # import time
    # t=time.time()
    if system_role == "":
        chat_history=[
                ChatMessage(role="system", content="Eres un asistente muy útil. Responde en español.")
                ]
    else:
        chat_history=[
                ChatMessage(role="system", content=f"Eres un {system_role.lower()}. Responde en español.")
                ]
    if len(session_list) > 0:
        for i in session_list:
            if i['role'] == 'user':
                role = MessageRole.USER
                chat_history.append(ChatMessage(role=role, content=i['content']))
            elif i['role'] == 'assistant':
                role = MessageRole.ASSISTANT
                chat_history.append(ChatMessage(role=role, content=i['content'], metadata=i['metadata']))
        return chat_history
    else:
        return chat_history
    
def check_exp_name(exp_name, session_string):
    exp_name_store = f'{exp_name}'
    if session_string != "":
        return True
    else:
        if exp_name_store in os.listdir('Storage'):
            st.error('El nombre de experimento ya existe, por favor elija otro.')
            return False
        elif (exp_name == ''):
            st.error('Ingrese un nombre de experimento.')
            return False
        else:
            return True
        
def check_role(system_role, exp_name=''):
    if system_role == "":
        return False
    else:
        save_personality(system_role, exp_name)
        return True
        
def save_personality(system_role, exp_name=""):
    try:
        os.makedirs(f'Storage/{exp_name}')
    except:
        pass
    f = open(f"Storage/{exp_name}/role.txt", "w")
    f.write(system_role)
    f.close()

def get_personality(exp_name):
    try:
        p = f'Storage/{exp_name}/role.txt'
        with open(p) as f:  
            contents = f.read()
        personality = contents.title()
        st.write(f'{8*"____"}\nLa personalidad del modelo es: {personality}')
    except:
        pass

def docs_uploaded(exp_name):
    try:
        os.makedirs(f'Storage/{exp_name}')
    except:
        pass

    docs = [i.removesuffix('.pdf') for i in os.listdir('temp')]
    string = 'El modelo fue entrenado con los siguientes PDFs:'
    for doc in docs:
        string += f'\n\n{doc}'
        
    f = open(f"Storage/{exp_name}/docs.txt", "w")
    f.write(string)
    f.close()

def get_docs_uploaded(exp_name):
    p = f'Storage/{exp_name}/docs.txt'
    with open(p) as f:  
        documents_uploaded = f.read()
    st.write(f'{8*"____"}\n{documents_uploaded}')
    

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