Agente Corporativo IA - Base de Conocimiento Interna

Este proyecto es un agente de Inteligencia Artificial conversacional diseñado para interactuar con los documentos internos de una empresa (E-commerce). Permite a cualquier colaborador hacer preguntas en lenguaje natural y recibir respuestas precisas basadas en los manuales y políticas de la organización, reduciendo el tiempo de búsqueda de información.

## 🏗️ Arquitectura del Proyecto

El sistema está construido utilizando un enfoque RAG (Retrieval-Augmented Generation) con las siguientes tecnologías:

*   **Lenguaje:** Python
*   **Orquestación:** LangChain
*   **Procesamiento de Documentos:** `PyPDFLoader` para la lectura y fragmentación del manual corporativo.
*   **Base de Conocimiento Vectorial:** FAISS (Facebook AI Similarity Search) para el almacenamiento y recuperación eficiente de fragmentos de texto.
*   **Embeddings y LLM:** Cohere (`embed-multilingual-v3.0` para la vectorización y el modelo base `command` para la generación de respuestas en lenguaje natural).

## 💬 Ejemplos de Preguntas y Respuestas

El agente es capaz de resolver dudas sobre diversas áreas como Recursos Humanos, Operaciones y Finanzas.

> **Usuario:** "Soy un nuevo empleado, ¿cuántos días de vacaciones tengo y cómo los pido?"
> **Agente:** Tienes derecho a 20 días hábiles de vacaciones remuneradas si tienes más de un año de antigüedad. Las solicitudes deben hacerse a través del portal "HR-Connect" con al menos 15 días de anticipación.

> **Usuario:** "¿Cuál es el límite de gastos para hospedaje en un viaje de trabajo?"
> **Agente:** El límite para gastos de hospedaje en un viaje de trabajo no debe superar los $150 USD diarios. Debes justificar el gasto subiendo las facturas al sistema "Expensify".

## 🚀 Instrucciones de Ejecución Local

Para correr este proyecto en tu propia máquina, sigue estos pasos:

1. Clona este repositorio.
2. Asegúrate de tener Python instalado.
3. Instala las dependencias ejecutando:
   `pip install -r requirements.txt`
4. Configura tu variable de entorno con tu API Key de Cohere dentro del archivo `app.py`.
5. Ejecuta la aplicación:
   `python app.py`

## ☁️ Deploy en Oracle Cloud Infrastructure (OCI)

A continuación se muestra la evidencia de la aplicación ejecutándose en un entorno de producción dentro de la nube de Oracle:

*(Nota: Aquí se insertará la imagen o enlace del deploy en OCI)*
