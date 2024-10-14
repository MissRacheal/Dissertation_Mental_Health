from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
# Adding the parent folder of "ml_chat_bot" to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ml_chat_model import MLChatModel

app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    incoming_message = request.json['message']

    conversational_chat = MLChatModel(prompt=incoming_message)
    ml_model_response = conversational_chat.run()

    return jsonify(reply=ml_model_response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)