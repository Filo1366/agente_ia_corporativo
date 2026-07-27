import os
import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

print("Cargando el PDF y creando la base de conocimiento...")
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
agente_rag_global = create_retrieval_chain(retriever, question_answer_chain)


@cl.on_chat_start
async def start():
    # Asignamos el agente global a la sesión, de forma inmediata y sin trabar el servidor
    cl.user_session.set("rag_chain", agente_rag_global)
    
    await cl.Message(
        content="¡Listo! Soy tu asistente corporativo. ¿Qué dudas tienes sobre las políticas de la empresa?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    try:
        rag_chain = cl.user_session.get("rag_chain")

        if not rag_chain:
            await cl.Message(content="⚠️ El agente no está disponible en este momento.").send()
            return

        response = rag_chain.invoke({"input": message.content})

        await cl.Message(content=response["answer"]).send()

    except Exception as e:
        await cl.Message(
            content=f"❌ Hubo un error al procesar tu solicitud: {str(e)}"
        ).send()
