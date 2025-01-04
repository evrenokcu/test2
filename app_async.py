import os
from dotenv import load_dotenv
from quart import Quart, request, jsonify
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini
from datetime import datetime
from quart_cors import cors
import aiohttp

# Load environment variables
#load_dotenv()

# Suppress gRPC warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

# Initialize Quart
app = Quart(__name__)
app = cors(app, allow_origin="*")
app.debug = True

# Initialize LLM clients
llms = {
    "ChatGPT": OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    "Claude": Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
    "Gemini": Gemini(api_key=os.getenv("GOOGLE_API_KEY")),
}

@app.get("/hello")
async def hello():
    return jsonify({"message": "Hello from Quart!"})

@app.post("/")
async def query_llm():
    """
    Query the specified LLM with a given prompt and return the response as a string.
    """
    data = await request.json
    llm_name = data.get("llm_name")
    prompt = data.get("prompt")

    # Check if the specified LLM is supported
    if llm_name not in llms:
        return jsonify({
            "error": f"LLM '{llm_name}' not supported. Available: {list(llms.keys())}"
        }), 400

    # Get the corresponding LLM client
    llm_client = llms[llm_name]

    # Query the LLM asynchronously
    try:
        response = await llm_client.acomplete(prompt)
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

@app.post("/direct_openai")
async def call_openai():
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = await client.chat.acompletions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Why is the sky blue?"}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return jsonify({
            "response": response.choices[0].message.content
        })
    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        })

@app.get("/test_connectivity")
async def test_connectivity():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://www.google.com") as response:
                return jsonify({"status": response.status, "message": "Internet connectivity is working"})
    except Exception as e:
        return jsonify({"error": f"Connectivity issue: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)