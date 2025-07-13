import numpy as np

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

from .modules import *

def calc_flops(model):
    @tf.function
    def model_fn(inputs):
        return model(inputs)

    input_data = tf.TensorSpec([1, *model.inputs[0].shape[1:]], model.inputs[0].dtype)
    graph = model_fn.get_concrete_function(input_data)
    graph = convert_variables_to_constants_v2_as_graph(graph)[0].graph

    identity = []

    total_flops = 0
    for op in graph.get_operations():
        outputs = op.outputs
        output = []
        
        if len(outputs):
            output = outputs[0].get_shape()

        if len(output) > 1:
            output = output[1:]

        if op.type == "Identity":
            identity = op.outputs[0].get_shape() # (kernel, kernel, in_channels, out_channels)

        elif op.type in ("Conv1D", "Conv2D", "Conv3D"):
            total_flops += np.prod([2, *identity, *output[:-1]], dtype=np.int64)
        
        elif op.type in ("AddV2", "Mul", "Sub", "Div", "Mean", "Maximum", "Square", "Sum", "Neg", "Sqrt", "BiasAdd"):
            total_flops += np.prod(output, dtype=np.int64)
        
        elif op.type in ("Rsqrt", "SquaredDifference"):
            total_flops += np.prod([2, *output], dtype=np.int64)
        
        elif op.type in ("Sigmoid", "Softmax"):
            total_flops += np.prod([3, *output], dtype=np.int64)
        
        elif op.type in ("Cast",):
            total_flops += np.prod(output, dtype=np.int64)
        
        elif op.type in ("MatMul",):
            total_flops += np.prod([2, *output], dtype=np.int64)
        
        elif op.type == "BatchMatMulV2":
            shape_a = op.inputs[0].shape
            shape_b = op.inputs[1].shape
            if None not in shape_a and None not in shape_b:
                total_flops += np.prod([2, *shape_a, shape_b[-1]], dtype=np.int64)
    
    return total_flops

def getitem(dataloader):
    return dataloader.__getitem__()

def progress_bar(it, its):
    return " %d/%d [%-16s]" % (it, its, "=" * round(((it + 1) / its) * 16))