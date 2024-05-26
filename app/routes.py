
from flask import request, jsonify
from .utils import load_onnx_model, save_onnx_model, edit_onnx_model, build_onnx_model

app = Flask(__name__)

@app.route('/load', methods=['POST'])
def load_model():
    model_path = request.json.get('model_path')
    try:
        model = load_onnx_model(model_path)
        return jsonify({"message": "Model loaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save', methods=['POST'])
def save_model():
    model_path = request.json.get('model_path')
    model = ...  # Load the model object from the session or database
    try:
        save_onnx_model(model, model_path)
        return jsonify({"message": "Model saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/edit', methods=['POST'])
def edit_model():
    node_name = request.json.get('node_name')
    new_value = np.array(request.json.get('new_value'), dtype=np.float32)
    model = ...  # Load the model object from the session or database
    try:
        model = edit_onnx_model(model, node_name, new_value)
        return jsonify({"message": "Model edited successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/build', methods=['POST'])
def build_model():
    input_dim = request.json.get('input_dim')
    output_dim = request.json.get('output_dim')
    try:
        model = build_onnx_model(input_dim, output_dim)
        return jsonify({"message": "Model built successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
