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
openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.api_type = st.secrets['OPENAI_API_TYPE']
openai.api_version = st.secrets['OPENAI_API_VERSION']
openai.api_base = st.secrets['OPENAI_API_BASE']

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
    
    vectorIndex.storage_context.persist(persist_dir = f'Store_{exp_name}')

    return vectorIndex

def load_index(exp_name=''):
    storage_context = StorageContext.from_defaults(persist_dir = f'Store_{exp_name}')
    index = load_index_from_storage(storage_context)

    return index

def call_agent(chat_history, exp_name=''):
    index=load_index(exp_name=exp_name)
    query_engine = index.as_query_engine()

    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine, 
        # condense_question_prompt=custom_prompt,
        chat_history=chat_history,
        verbose=False
    )
    return chat_engine

def get_history(session_list):
    # import time
    # t=time.time()
    chat_history=[
            ChatMessage(role="system", content="You are a very useful assistant.")
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
    
#TODO LIST:
# 1. Obtener la metadata de las respuestas -> done
# 2. Que funcione con credenciales de azure -> not done
# 3. Desplegar online -> done
# 4. Modulo para subir resultados. -> done
# 5. personalizas system.



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