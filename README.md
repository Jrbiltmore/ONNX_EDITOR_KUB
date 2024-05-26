
# ONNX Model Editor

This project provides a web interface and CLI for loading, editing, saving, and building ONNX models.

## Setup

1. Clone the repository.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask app:

```bash
flask run
```

## Docker

To build and run the Docker container:

```bash
docker build -t onnx_model_editor .
docker run -p 5000:5000 onnx_model_editor
```
