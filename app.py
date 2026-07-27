import os
from unittest import loader
from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def iniciar_agente():
    os.environ["COHERE_API_KEY"] = "hj8JcUZexwBmCpq68X2EajhGLM5gin5rZUuH9sNZ"

    print("Cargando y leyendo el documento...")

    try:
        loader = PyPDFLoader("manual_corporativo_ecommerce.pdf")
        documentos = loader.load_and_split()
    except Exception as e:
        print(f"Error al cargar el documento: {e}")
        return

    print("Construyendo la base del conocimiento...")
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    vectorstore = FAISS.from_documents(documentos, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatCohere(model="command-r-08-2024")

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

    agente = create_retrieval_chain(retriever, create_stuff_documents_chain(llm,prompt))
    print("✅ ¡Agente listo!\n")

    print("Escribe salir para terminar la conversación.\n")
    while True:
        pregunta = input("Pregunta: ")
        if pregunta.lower() == "salir":
            print("👋 ¡Hasta luego!")
            break

        respuesta = agente.invoke({"input": pregunta})
        print(f"\n🤖 Respuesta del Agente:\n{respuesta['answer']}")

if __name__ == "__main__":
    iniciar_agente()
