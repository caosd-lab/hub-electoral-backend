import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# --- Configuración y Carga ---
app = Flask(__name__)
CORS(app)

print("Cargando los modelos de IA y la base de conocimiento...")

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("la variable de entorno GOOGLE_API_KEY no está configurada.")

# Cargamos los dos motores de IA
llm_flash = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.1)
llm_pro = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)

# Cargamos nuestra "enciclopedia" (el archivo JSON) en la memoria
try:
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
    print("Base de conocimiento 'knowledge_base.json' cargada exitosamente.")
except FileNotFoundError:
    knowledge_base = []
    print("ADVERTENCIA: No se encontró 'knowledge_base.json'. El asistente no tendrá información.")

# --- Creamos las Cadenas de Pensamiento ---
# (Las definiciones de PROMPT_CLASSIFIER, classifier_chain, final_response_prompt, final_chain no cambian)
# 1. El "Recepcionista" (Clasificador de Intención)
prompt_classifier_template = """
Clasifica el siguiente texto del usuario en una de estas tres categorías: 'saludo', 'charla_general' o 'pregunta_analitica'.
- 'saludo': Para saludos simples como 'hola', 'buenos días'.
- 'charla_general': Para preguntas sobre tus capacidades, tu estado o temas fuera de los programas (ej: '¿cómo puedes ayudarme?', '¿qué haces?', '¿cómo estás?').
- 'pregunta_analitica': Para cualquier pregunta que requiera analizar el contenido de los programas electorales.

Responde únicamente con una de esas tres clasificaciones.

TEXTO DEL USUARIO: "{user_input}"
CLASIFICACIÓN:
"""
PROMPT_CLASSIFIER = PromptTemplate(template=prompt_classifier_template, input_variables=["user_input"])
classifier_chain = LLMChain(llm=llm_flash, prompt=PROMPT_CLASSIFIER)

# 2. El "Analista Final" (con el Manual de Estilo Restaurado)
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
    - Importante: No incluyas los marcadores de bloque de código como ```html. Tu respuesta debe empezar directamente con la primera etiqueta HTML.

    --- BASE DE CONOCIMIENTO ESTRUCTURADA ---
    {contexto_preciso}
    --- FIN DE LA BASE DE CONOCIMIENTO ---

    PREGUNTA DEL USUARIO:
    {pregunta_usuario}

    RESPUESTA PROFESIONAL Y BIEN FORMATEADA (siguiendo la Regla 2):
    """
)
final_chain = LLMChain(llm=llm_pro, prompt=final_response_prompt)


print("¡Servidor de consulta (Maestro) listo!")

@app.route('/ask', methods=['POST'])
def ask_question():
    pregunta = request.json.get('question', '')
    if not pregunta:
        return jsonify({"error": "No se proporcionó ninguna pregunta."}), 400

    print(f"Recibida pregunta: '{pregunta}'")
    try:
        # <<< INICIO DE LA MODIFICACIÓN PARA PRUEBA >>>
        # --- PASO 1: DETECCIÓN DE INTENCIÓN (Temporalmente desactivado para prueba) ---
        # Comentamos la llamada a la IA para clasificar la intención
        # intent_response = classifier_chain.invoke({"user_input": pregunta})
        # intent = intent_response['text'].strip().lower()
        # print(f"Intención detectada: {intent}")

        # En su lugar, hacemos una comprobación simple y directa para saludos comunes
        pregunta_limpia = pregunta.lower().strip()
        if pregunta_limpia in ["hola", "hi", "buenos dias", "buenas tardes", "buenas noches", "hello"]:
             intent = "saludo"
             print(f"Intención detectada (simple): {intent}") # Añadimos un print para saber que funcionó
        else:
             # Si no es un saludo simple, asumimos que es una pregunta analítica para esta prueba
             intent = "pregunta_analitica"
             print(f"Intención detectada (simple): {intent}")
        # <<< FIN DE LA MODIFICACIÓN PARA PRUEBA >>>

        if "saludo" in intent:
            # Esta respuesta ahora es inmediata, no requiere llamada a la IA
            respuesta_texto = "¡Hola! Soy el asistente del Hub Electoral. ¿En qué puedo ayudarte con los programas presidenciales?"
            return jsonify({"answer": respuesta_texto, "sources": []})
        
        # El manejo de 'charla_general' queda temporalmente inactivo con este cambio,
        # ya que ahora todo lo que no es saludo se trata como 'pregunta_analitica'.
        # Esto es solo para la prueba.
        
        # Si la intención es 'pregunta_analitica', procedemos como antes
        else: # (intent == "pregunta_analitica")
            contexto = json.dumps(knowledge_base, ensure_ascii=False, indent=2)
            
            respuesta_final = final_chain.invoke({
                "contexto_preciso": contexto,
                "pregunta_usuario": pregunta
            })

            fuentes = [candidato.get('candidato_nombre', 'Candidato sin nombre') for candidato in knowledge_base]
            return jsonify({"answer": respuesta_final['text'], "sources": fuentes})

    except Exception as e:
        print(f"Error en el servidor: {e}")
        return jsonify({"error": f"Error interno del servidor: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
