from flask import Flask, request, jsonify
from utils import query_data
from flasgger import Swagger
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
swagger = Swagger(app)



@app.route('/query/<prompt>', methods=['GET'])
def query_prompt(prompt):
    if prompt:
        response = query_data(prompt)
        return jsonify({'response': response}), 200
    else:
        return jsonify({'error': 'Invalid request. Prompt not provided.'}), 400
    
if __name__ == '__main__':
   app.run()