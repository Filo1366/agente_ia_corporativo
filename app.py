import os
import streamlit as st
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Asistente Corporativo", page_icon="💼")

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("💼 Asistente de Políticas de la Empresa")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy tu asistente corporativo. ¿Qué dudas tienes sobre las políticas?"}
    ]

@st.cache_resource
def iniciar_agente():
    COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
    

    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=COHERE_API_KEY)

    vectorstore = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever()

    llm = ChatCohere(model="command-r-08-2024", cohere_api_key=COHERE_API_KEY)

    system_prompt = (
        "Eres un asistente corporativo de inteligencia artificial. "
        "Responde las preguntas de los colaboradores utilizando ÚNICAMENTE la información provista en el contexto. "
        "Si la respuesta no está en el contexto, di amablemente que no tienes esa información.\n\n"
        "Contexto:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu asistente corporativo. ¿Qué dudas tienes sobre las políticas?"}]

try:
    rag_chain = iniciar_agente()
except Exception as e:
    st.error(f"⚠️ Error al cargar el agente: {e}")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    # Mostramos la pregunta del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en el manual corporativo..."):
            try:
                response = rag_chain.invoke({"input": prompt})
                respuesta_texto = response["answer"]
                st.markdown(respuesta_texto)
                # Guardamos la respuesta en el historial
                st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})
            except Exception as e:
                st.error(f"❌ Hubo un error: {str(e)}")
