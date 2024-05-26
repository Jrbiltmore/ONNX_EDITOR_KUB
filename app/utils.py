
import onnx
import numpy as np
from onnx import helper, shape_inference

def load_onnx_model(model_path: str) -> onnx.ModelProto:
    model = onnx.load(model_path)
    return model

def save_onnx_model(model: onnx.ModelProto, model_path: str):
    onnx.save(model, model_path)

def edit_onnx_model(model: onnx.ModelProto, node_name: str, new_value: np.ndarray) -> onnx.ModelProto:
    for node in model.graph.node:
        if node.name == node_name:
            for attr in node.attribute:
                if attr.name == 'value':
                    attr.t.raw_data = new_value.tobytes()
                    attr.t.dims[:] = new_value.shape
                    attr.t.data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[new_value.dtype]
    return model

def build_onnx_model(input_dim: int, output_dim: int) -> onnx.ModelProto:
    X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [None, input_dim])
    Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [None, output_dim])

    node = helper.make_node(
        'Add',
        inputs=['X', 'W'],
        outputs=['Y']
    )

    W = helper.make_tensor(
        'W',
        onnx.TensorProto.FLOAT,
        [input_dim, output_dim],
        np.random.rand(input_dim, output_dim).astype(np.float32)
    )

    graph = helper.make_graph(
        [node],
        'simple_graph',
        [X],
        [Y],
        [W]
    )

    model = helper.make_model(graph, producer_name='onnx-example')
    model = shape_inference.infer_shapes(model)
    return model
