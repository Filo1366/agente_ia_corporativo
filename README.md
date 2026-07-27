# 💼 Asistente Corporativo de Inteligencia Artificial (RAG)

## 📌 Descripción General del Proyecto
Este proyecto consiste en un Agente Inteligente basado en la arquitectura **RAG (Retrieval-Augmented Generation)** diseñado para asistir a los colaboradores de una empresa de e-commerce. El agente responde preguntas frecuentes sobre políticas internas, recursos humanos, finanzas y logistica.
---

## 🏗️ Arquitectura de la Solución
La solución implementa un flujo RAG optimizado para la nube:
1. **Carga y Procesamiento:** El documento PDF (`manual_corporativo_ecommerce.pdf`) se divide en fragmentos manejables mediante `PyPDFLoader`.
2. **Vectorización:** Los fragmentos de texto se convierten en vectores densos utilizando los modelos multilingües de **Cohere Embeddings**.
3. **Almacenamiento Vectorial:** Se utiliza una base de datos vectorial local **FAISS** (`faiss_index`) para realizar búsquedas semánticas eficientes.
4. **Generación de Respuestas:** El motor de lenguaje **Cohere Command-R** recibe la consulta del usuario junto con el contexto recuperado del manual para generar una respuesta precisa y acotada.
5. **Interfaz de Usuario:** Aplicación web interactiva desarrollada en **Streamlit** y desplegada en la nube.

---

## 🛠️ Tecnologías y Herramientas Utilizadas
* **Lenguaje:** Python 3.10+
* **Orquestación de IA:** LangChain (`langchain-classic`, `langchain-community`)
* **Modelo de Lenguaje y Embeddings:** Cohere API (`embed-multilingual-v3.0` y `command-r-08-2024`)
* **Base de Datos Vectorial:** FAISS-CPU
* **Interfaz Gráfica:** Streamlit
* **Control de Versiones y Despliegue:** Git, GitHub y Streamlit Community Cloud

---

## ⚙️ Instrucciones para Ejecutar el Proyecto Localmente

1. Clona este repositorio en tu computadora:
   ```bash
   git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
   cd tu-repositorio
2. Instala las dependencias necesarias:
   pip install -r requirements.txt
3. Configura tu variable de entorno con tu API Key de Cohere:
   En Linux/Mac: export COHERE_API_KEY="tu_api_key"
   En Windows (PowerShell): $env:COHERE_API_KEY="tu_api_key"
4. Ejecuta la aplicación de Streamlit:
   streamlit run app.py
---

💬 **Ejemplos de Preguntas que el Agente Puede Responder**
¿Cuántos días de vacaciones me corresponden si tengo más de un año en la empresa?  
¿Cuál es el límite permitido para viáticos de alimentación en viajes de negocio?  
¿Qué debo hacer si recibo un correo sospechoso pidiendo credenciales?  
¿Cómo se manejan los productos devueltos que presentan daños menores al 10%?  

---

🤖 **Ejemplos de Respuestas Generadas por el Agente**
Pregunta: ¿Cuál es la política de trabajo remoto y presencial?  
Respuesta: La empresa opera bajo un modelo híbrido flexible. Los colaboradores deben asistir a la oficina principal un mínimo de 2 días a la semana, permitiendo el trabajo remoto los días restantes.  

---

☁️ **Evidencia del Deploy y Funcionamiento**

1. Interfaz del Asistente en la Nube  

2. Estructura de Archivos del Repositorio  
