import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini
from datetime import datetime
from pydantic import BaseModel
# Load environment variables
# env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
# load_dotenv(dotenv_path=env_path)

# Suppress gRPC warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

# Initialize Flask
app = Flask(__name__)

# Initialize LLM clients
llms = {
    "ChatGPT": OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    "Claude": Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
    "Gemini": Gemini(api_key=os.getenv("GOOGLE_API_KEY")),
}
class EchoData(BaseModel):
    message: str

@app.get("/hello")
def hello():
    return {"message": "Hello from FastAPI!"}

@app.post("/echo")
async def echo(data: EchoData):
    return {"you_sent": data.dict()}

@app.route("/", methods=["POST"])
def query_llm():
    """
    Query the specified LLM with a given prompt and return the response as a string.
    """
    data = request.json
    llm_name = data.get("llm_name")
    prompt = data.get("prompt")

    # Check if the specified LLM is supported
    if llm_name not in llms:
        return jsonify({
            "error": f"LLM '{llm_name}' not supported. Available: {list(llms.keys())}"
        }), 400
    
    # Get the corresponding LLM client
    llm_client = llms[llm_name]

    # Query the LLM
    try:
        response = llm_client.acomplete(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)
        return jsonify({
            "llm": llm_name,
            "response": response_text,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })
    except Exception as e:
        return jsonify({
            "error": f"Error querying {llm_name}: {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
