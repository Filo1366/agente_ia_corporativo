import os
import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def construir_agente():
    COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
    
    loader = PyPDFLoader("manual_corporativo_ecommerce.pdf")
    documentos = loader.load_and_split()
    
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=COHERE_API_KEY)
    vectorstore = FAISS.from_documents(documentos, embeddings)
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

@cl.on_chat_start
async def start():
    await cl.Message(
        content="¡Hola! Soy tu asistente corporativo. Escribe 'Hola' para que comience a leer el manual y podamos platicar."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")

    if not rag_chain:
        msg = cl.Message(content="⏳ Preparando el documento por primera vez... Dame unos 15 segundos.")
        await msg.send()
        
        try:
            rag_chain = construir_agente()
            cl.user_session.set("rag_chain", rag_chain)
            
            msg.content = "✅ ¡Documento listo! Procesando tu pregunta..."
            await msg.update()
            
        except Exception as e:
            msg.content = f"❌ Error al cargar el documento: {str(e)}"
            await msg.update()
            return

    try:
        response = rag_chain.invoke({"input": message.content})
        await cl.Message(content=response["answer"]).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ Hubo un error al generar la respuesta: {str(e)}"
        ).send()
