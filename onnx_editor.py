
import argparse
import numpy as np
from app.utils import load_onnx_model, save_onnx_model, edit_onnx_model, build_onnx_model

def main():
    parser = argparse.ArgumentParser(description='ONNX Model Editor and Builder')
    parser.add_argument('--load', type=str, help='Path to load the ONNX model')
    parser.add_argument('--save', type=str, help='Path to save the ONNX model')
    parser.add_argument('--edit-node', type=str, help='Name of the node to edit')
    parser.add_argument('--new-value', type=str, help='New value for the node (comma-separated)')
    parser.add_argument('--build', action='store_true', help='Flag to build a new ONNX model')
    parser.add_argument('--input-dim', type=int, help='Input dimension for the new model')
    parser.add_argument('--output-dim', type=int, help='Output dimension for the new model')

    args = parser.parse_args()

    if args.load:
        model = load_onnx_model(args.load)

    if args.edit_node and args.new_value:
        new_value = np.fromstring(args.new_value, sep=',').reshape((2, 2))  # Adjust shape as necessary
        model = edit_onnx_model(model, args.edit_node, new_value)

    if args.save:
        save_onnx_model(model, args.save)

    if args.build:
        if args.input_dim is not None and args.output_dim is not None:
            model = build_onnx_model(args.input_dim, args.output_dim)
            if args.save:
                save_onnx_model(model, args.save)
        else:
            print("Input and output dimensions are required to build a new model.")

if __name__ == "__main__":
    main()
