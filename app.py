import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# GET endpoint
@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello from Flask!"})

# POST endpoint
@app.route('/echo', methods=['POST'])
def echo():
    data = request.json
    return jsonify({"you_sent": data})

if __name__ == '__main__':
    # Cloud Run sets the PORT environment variable; default to 8080 if not set
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
