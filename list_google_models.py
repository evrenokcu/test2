import google.generativeai as generativeai

# Authenticate with your API key
generativeai.auth = "AIzaSyAyvQKC3cvdVFkbczIyQnyU4bIw8mAiXUk"

# List available models
models = generativeai.list_models()
for model in models:
    print(f"Model ID: {model.name}, ")