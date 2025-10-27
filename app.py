import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
# <<< CORRECCIÓN: Importamos PromptTemplate desde langchain_core >>>
from langchain_core.prompts import PromptTemplate
# <<< CORRECCIÓN: Ya no necesitamos LLMChain si usamos la sintaxis | >>>
# from langchain.chains.llm import LLMChain

# --- Configuración y Carga ---
app = Flask(__name__)
CORS(app)

print("Cargando los modelos de IA y la base de conocimiento...")

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("la variable de entorno GOOGLE_API_KEY no está configurada.")

# Cargamos los dos motores de IA
# <<< CORRECCIÓN: Usamos el nombre de modelo estable sin -latest para flash >>>
llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
llm_pro = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)

# Cargamos nuestra "enciclopedia" (el archivo JSON) en la memoria
try:
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
    print("Base de conocimiento 'knowledge_base.json' cargada exitosamente.")
except FileNotFoundError:
    knowledge_base = []
    print("ADVERTENCIA: No se encontró 'knowledge_base.json'. El asistente no tendrá información.")

# --- Creamos las Cadenas de Pensamiento ---

# <<< INICIO DE LA MODIFICACIÓN 1: El "Recepcionista" ahora es más inteligente >>>
# 1. El "Recepcionista" (Clasificador de Intención)
# Añadimos la categoría 'charla_general' y actualizamos la descripción.
prompt_classifier_template = """
Clasifica el siguiente texto del usuario en una de estas tres categorías: 'saludo', 'charla_general' o 'pregunta_analitica'.
- 'saludo': Para saludos simples como 'hola', 'buenos días'.
- 'charla_general': Para preguntas sobre tus capacidades, tu estado o temas fuera de los programas electorales (ej: '¿cómo puedes ayudarme?', '¿qué haces?', '¿cómo estás?').
- 'pregunta_analitica': Para cualquier pregunta que requiera analizar el contenido de los programas electorales.

Responde únicamente con una de esas tres clasificaciones.

TEXTO DEL USUARIO: "{user_input}"
CLASIFICACIÓN:
"""
PROMPT_CLASSIFIER = PromptTemplate(template=prompt_classifier_template, input_variables=["user_input"])
# <<< Usamos la sintaxis moderna para crear la cadena >>>
classifier_chain = PROMPT_CLASSIFIER | llm_flash
# <<< FIN DE LA MODIFICACIÓN 1 >>>


# <<< INICIO DE LA MODIFICACIÓN 2: El "Manual de Estilo" ahora incluye tablas con bordes >>>
# 2. El "Analista Final" (con el Manual de Estilo Restaurado y Mejorado)
final_response_prompt = PromptTemplate(
    input_variables=["contexto_preciso", "pregunta_usuario"],
    template="""
    Tu tarea es actuar como un analista político experto. Usando la siguiente información ya extraída y estructurada de los programas, responde la pregunta del usuario.

    Regla 1: Análisis Exhaustivo.
    - Basa tu respuesta únicamente en la información contenida en la base de conocimiento proporcionada. Si la información no se encuentra, indícalo claramente.
    - Responde siempre en idioma español.

    Regla 2: Instrucciones de Formato de Salida en HTML.
    - Tu respuesta final debe ser generada como código HTML simple y limpio.
    - Para resaltar texto o títulos, usa etiquetas <b>...</b>.
    - Para listas, utiliza <ul> y <li>.
    - Si se pide una tabla, genera una tabla HTML bien estructurada. Añade estilos directamente a las etiquetas para que tenga bordes y sea legible.
      Ejemplo de una tabla bien formateada:
      <table style="width:100%; border-collapse: collapse; border: 1px solid #555;">
        <thead>
          <tr style="background-color: #444;">
            <th style="padding: 8px; border: 1px solid #555; text-align: left;">Candidato</th>
            <th style="padding: 8px; border: 1px solid #555; text-align: left;">Propuesta Clave</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style="padding: 8px; border: 1px solid #555;">Nombre Candidato</td>
            <td style="padding: 8px; border: 1px solid #555;">Detalle de la propuesta.</td>
          </tr>
          </tbody>
      </table>
    - Importante: No incluyas los marcadores de bloque de código como ```html. Tu respuesta debe empezar directamente con la primera etiqueta HTML (ej: <p>, <ul>, <table>) y terminar con la última etiqueta de cierre correspondiente.

    --- BASE DE CONOCIMIENTO ESTRUCTURADA ---
    {contexto_preciso}
    --- FIN DE LA BASE DE CONOCIMIENTO ---

    PREGUNTA DEL USUARIO:
    {pregunta_usuario}

    RESPUESTA PROFESIONAL Y BIEN FORMATEADA (siguiendo la Regla 2):
    """
)
# <<< Usamos la sintaxis moderna para crear la cadena >>>
final_chain = final_response_prompt | llm_pro
# <<< FIN DE LA MODIFICACIÓN 2 >>>


print("¡Servidor de consulta (Maestro) listo!")

@app.route('/ask', methods=['POST'])
def ask_question():
    pregunta = request.json.get('question', '')
    if not pregunta:
        return jsonify({"error": "No se proporcionó ninguna pregunta."}), 400

    print(f"Recibida pregunta: '{pregunta}'")
    try:
        # --- PASO 1: DETECCIÓN DE INTENCIÓN ---
        # <<< Usamos .invoke() y accedemos a .content >>>
        intent_response = classifier_chain.invoke({"user_input": pregunta})
        intent = intent_response.content.strip().lower()
        print(f"Intención detectada: {intent}")

        if "saludo" in intent:
            respuesta_texto = "¡Hola! Soy el asistente del Hub Electoral. ¿En qué puedo ayudarte con los programas presidenciales?"
            return jsonify({"answer": respuesta_texto, "sources": []})
        
        # <<< INICIO DE LA MODIFICACIÓN 1 (continuación): Manejo de "charla_general" >>>
        elif "charla_general" in intent:
            # Para estas preguntas, creamos una respuesta amigable al vuelo.
            conversational_prompt = PromptTemplate(
                input_variables=["pregunta_usuario"],
                template="""
                Eres un asistente de IA llamado Hub Electoral. Tu única función es analizar programas de gobierno de candidatos.
                Responde a la siguiente pregunta del usuario de forma breve, amigable y en español, recordando siempre cuál es tu propósito principal.
                Pregunta: {pregunta_usuario}
                Respuesta:
                """
            )
            # <<< Usamos la sintaxis moderna para crear y llamar la cadena >>>
            conversational_chain = conversational_prompt | llm_flash
            response = conversational_chain.invoke({"pregunta_usuario": pregunta})
            # <<< Accedemos a .content >>>
            return jsonify({"answer": response.content, "sources": []})
        # <<< FIN DE LA MODIFICACIÓN 1 >>>
        
        # Si la intención es 'pregunta_analitica', procedemos como antes
        else:
            contexto = json.dumps(knowledge_base, ensure_ascii=False, indent=2)
            
            respuesta_final = final_chain.invoke({
                "contexto_preciso": contexto,
                "pregunta_usuario": pregunta
            })

            fuentes = [candidato.get('candidato_nombre', 'Candidato sin nombre') for candidato in knowledge_base]
            # <<< Accedemos a .content >>>
            return jsonify({"answer": respuesta_final.content, "sources": fuentes})

    except Exception as e:
        print(f"Error en el servidor: {e}")
        return jsonify({"error": f"Error interno del servidor: {e}"}), 500

if __name__ == '__main__':
    # El calentamiento no es necesario con esta arquitectura estable
    app.run(host='0.0.0.0', port=8080)
