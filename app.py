import os
import chainlit as cl
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def iniciar_agente():
    COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
    
    # 1. Cargamos el modelo de embeddings
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=COHERE_API_KEY)
    
    # 2. MAGIA: Cargamos la base de datos pre-calculada en lugar de leer el PDF
    vectorstore = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True # Requisito de seguridad de FAISS
    )
    retriever = vectorstore.as_retriever()

    # 3. Configuramos el cerebro del chat
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

@cl.on_chat_start
async def start():
    try:
        rag_chain = iniciar_agente()
        cl.user_session.set("rag_chain", rag_chain)
        
        await cl.Message(
            content="¡Listo! Soy tu asistente corporativo. ¿Qué dudas tienes sobre las políticas de la empresa?"
        ).send()
        
    except Exception as e:
        await cl.Message(content=f"⚠️ Error al iniciar el agente: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    try:
        rag_chain = cl.user_session.get("rag_chain")

        if not rag_chain:
            await cl.Message(content="⚠️ El agente no está disponible.").send()
            return

        response = rag_chain.invoke({"input": message.content})
        await cl.Message(content=response["answer"]).send()

    except Exception as e:
        await cl.Message(content=f"❌ Hubo un error: {str(e)}").send()
