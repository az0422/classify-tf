import numpy as np

import tensorflow as tf

def calc_flops(model):
    @tf.function
    def model_fn(inputs):
        return model(inputs)

    input_data = tf.TensorSpec([1, *model.inputs[0].shape[1:]], model.inputs[0].dtype)
    graph = model_fn.get_concrete_function(input_data).graph

    total_flops = 0
    for op in graph.get_operations():
        outputs = op.outputs
        output = []
        
        if len(outputs):
            output = outputs[0].get_shape()

        if len(output) > 1:
            output = output[1:]
        elif len(output) == 0:
            output = [1]

        if op.type in ("Conv1D", "Conv2D", "Conv3D"):
            input_shape = op.inputs[1].get_shape()
            total_flops += np.prod([2, *input_shape, *output[:-1]], dtype=np.int64)
        
        elif op.type in ("AddV2", "Mul", "Sub", "Div", "Mean", "Maximum", "Square", "Sum", "Neg", "Sqrt", "BiasAdd"):
            total_flops += np.prod(output, dtype=np.int64)
        
        elif op.type in ("Rsqrt", "SquaredDifference"):
            total_flops += np.prod([2, *output], dtype=np.int64)
        
        elif op.type in ("Sigmoid", "Softmax"):
            total_flops += np.prod([3, *output], dtype=np.int64)
        
        elif op.type in ("MatMul",):
            total_flops += np.prod([2, *output], dtype=np.int64)
        
        elif op.type == "BatchMatMulV2":
            shape_a = op.inputs[0].shape
            shape_b = op.inputs[1].shape
            if None not in shape_a and None not in shape_b:
                total_flops += np.prod([2, *shape_a, shape_b[-1]], dtype=np.int64)

        elif op.type in ("Cast", "Placeholder", "Identity", "IdentityN", "ReadVariableOp", "Const", "Relu"):
            total_flops += 0
    
    return total_flops