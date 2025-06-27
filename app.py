import os
import glob
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.llm import LLMChain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)
CORS(app)

print("Cargando los modelos de IA...")
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("la variable de entorno GOOGLE_API_KEY no está configurada.")

llm_flash = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
llm_pro = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)

analyst_prompt_template = """
Tu tarea es actuar como un analista político experto. A continuación se te entregan varios documentos, cada uno claramente delimitado por marcadores de INICIO y FIN. Debes seguir estas reglas rigurosamente:

Regla 1: Análisis Exhaustivo.
- Lee CADA UNO de los documentos proporcionados para formular tu respuesta. Tu principal objetivo es sintetizar la información de TODAS las fuentes.
- Basa tu respuesta únicamente en la información contenida dentro de los documentos. Si la información no se encuentra, indícalo claramente.
- Responde siempre en idioma español.

Regla 2: Instrucciones de Formato de Salida en HTML.
- Tu respuesta final debe ser generada como código HTML simple y limpio.
- Para resaltar texto o para títulos, usa etiquetas de negrita (<b>ejemplo</b>).
- Para presentar listas de puntos, utiliza etiquetas de lista HTML (<ul><li>Punto 1</li><li>Punto 2</li></ul>).
- Si se pide una tabla, genera una tabla HTML con <table>, <tr>, <th> y <td>.
- **Importante: No incluyas los marcadores de bloque de código como ```html al principio o ``` al final. Tu respuesta debe empezar directamente con la primera etiqueta HTML (ej: <b>, <ul>, o <table>) y terminar con la última etiqueta de cierre.**

--- DOCUMENTOS PROPORCIONADOS ---
{documentos}
--- FIN DE LOS DOCUMENTOS ---

PREGUNTA DEL USUARIO:
{pregunta_usuario}

RESPUESTA EN CÓDIGO HTML (siguiendo la Regla 2):
"""
ANALYST_PROMPT = PromptTemplate(template=analyst_prompt_template, input_variables=["documentos", "pregunta_usuario"])

print("Precargando y estructurando todos los documentos fuente...")
textos_estructurados = []
fuentes_precargadas = []
for filepath in glob.glob('fuentes/*.pdf'):
    filename = os.path.basename(filepath)
    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()
    contenido_completo_pdf = "\n".join([page.page_content for page in pages])
    bloque_documento = f"--- INICIO DEL DOCUMENTO: {filename} ---\n{contenido_completo_pdf}\n--- FIN DEL DOCUMENTO: {filename} ---"
    textos_estructurados.append(bloque_documento)
    fuentes_precargadas.append(filename)
contexto_global_estructurado = "\n\n".join(textos_estructurados)
print(f"Se han precargado y estructurado {len(fuentes_precargadas)} documentos.")

print("¡Servidor listo!")

@app.route('/ask', methods=['POST'])
def ask_question():
    json_data = request.get_json()
    pregunta = json_data.get('question', '')
    if not pregunta:
        return jsonify({"error": "No se proporcionó ninguna pregunta."}), 400

    palabra_clave_pro = "analiza:"
    pregunta_lower = pregunta.lower().strip()
    motor_seleccionado = llm_flash
    
    if pregunta_lower.startswith(palabra_clave_pro):
        motor_seleccionado = llm_pro
        pregunta = pregunta[len(palabra_clave_pro):].strip()
        print(f"Recibida pregunta para MODO EXPERTO: '{pregunta}'")
    else:
        print(f"Recibida pregunta para MODO RÁPIDO: '{pregunta}'")

    cadena_analista = LLMChain(prompt=ANALYST_PROMPT, llm=motor_seleccionado)
    
    try:
        respuesta_texto = cadena_analista.invoke({
            "documentos": contexto_global_estructurado,
            "pregunta_usuario": pregunta
        })['text']
        return jsonify({"answer": respuesta_texto, "sources": fuentes_precargadas})
    except Exception as e:
        print(f"Error en Modo Analista: {e}")
        return jsonify({"error": f"Error interno del servidor: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)