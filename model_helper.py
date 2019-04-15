import numpy as np
import tensorflow as tf

from collections import namedtuple


class TrainModel(namedtuple(
    "TrainModel", ("graph", "model"))):
    pass

class InferModel(namedtuple(
    "InferModel", ("graph", "model"))):
    pass


def flatten(inputs):
    """Flatten inputs which is the output of the convolution operation."""
    # [batch, height, width, channels]
    inp_shape = inputs.get_shape().as_list()
    assert len(inp_shape) == 4 
    
    prod_val = np.prod(inp_shape[1:])
    # [batch, height*width*channels]
    inputs = tf.reshape(inputs, [-1, prod_val]) 
    return inputs

def linear(inputs, num_outputs, w_init, b_init, activation_fn=None, scope='linear'):
    """Make fully connected layer."""
    inp_shape = inputs.get_shape().as_list()
    assert len(inp_shape) == 2
    
    with tf.variable_scope(scope):
        weights = tf.get_variable(
            'w', [inp_shape[-1], num_outputs], initializer=w_init)
        biases = tf.get_variable(
            'b', [num_outputs], initializer=b_init)
        output = tf.add(tf.matmul(inputs, weights), biases)

        if activation_fn:
            output = activation_fn(output)
            
        return output

def conv2d(inputs, num_outputs, f_height, f_width, w_init, b_init, stride, 
           padding="SAME", activation_fn=tf.nn.relu, scope="conv2d"):
    """Make convolution layer."""
    # [batch, height, width, channels]
    inp_shape = inputs.get_shape().as_list()
    assert len(inp_shape) == 4
    
    with tf.variable_scope(scope):
        filter_w = tf.get_variable('w', 
            [f_height, f_width, inp_shape[-1], num_outputs],
            initializer=w_init)
        filter_b = tf.get_variable('b', 
            [num_outputs], initializer=b_init)
        strides = [1, stride, stride, 1]
        
        output = tf.nn.conv2d(
            inputs, filter_w, strides, padding)
        output = tf.nn.bias_add(
            output, filter_b)
        
        if activation_fn:
            output = activation_fn(output)
        
        return output

def single_cell(cell_type, num_units, dropout, mode, 
                residual_connect=False, residual_fn=None, device_str=None):
    """Create a single RNN cell."""
    # Dropout(=1.0 - keep_prob) 
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
    
    # Cell type
    if cell_type == "lstm":
        single_cell = tf.nn.rnn_cell.LSTMCell(num_units)
    else:
        raise ValueError("Unknown cell type %s!" % (cell_type))
    
    # Dropout
    if dropout:
        single_cell = tf.nn.rnn_cell.DropoutWrapper(
            single_cell, input_keep_prob=(1.0 - dropout))
    # Residual 
    if residual_connect:
        single_cell = tf.nn.rnn_cell.ResidualWrapper(
            single_cell, residual_fn=residual_fn)
    # Device wrapper
    if device_str:
        single_cell = tf.nn.rnn_cell.DeviceWrapper(
            single_cell, device_str)
        
    return single_cell
    
def cell_list(num_layers, cell_type, num_units, dropout, mode, 
              residual_connect=False, residual_fn=None, num_gpus=0):
    """Create a list of RNN cells."""
    cell_list = []
    for i in range(num_layers):
        cell = single_cell(
            cell_type=cell_type, 
            num_units=num_units, 
            dropout=dropout, 
            mode=mode, 
            residual_connect=residual_connect, 
            residual_fn=residual_fn)
        
        cell_list.append(cell)
        
    return cell_list

def create_rnn_cell(num_layers, cell_type, num_units, dropout, mode, 
                    residual_connect=False, residual_fn=None, num_gpus=0):
    """Create multi-layer RNN cell."""
    rnn_cell_list = cell_list(
        num_layers=num_layers, 
        cell_type=cell_type, 
        num_units=num_units, 
        dropout=dropout, 
        mode=mode, 
        residual_connect=residual_connect, 
        residual_fn=residual_fn, 
        num_gpus=num_gpus)

    # Single layer
    if len(rnn_cell_list) == 1:
        return rnn_cell_list[0]
    else:
        return tf.nn.rnn_cell.MultiRNNCell(rnn_cell_list)

def get_initializer(init_op, seed, mean=0.0, stddev=0.1, bias_start=0.0):
    """Get initializer for weights, biases."""
    if init_op == 'truncated_normal':
        initializer = tf.initializers.truncated_normal(
            mean=mean, stddev=stddev, seed=seed)
    elif init_op == 'glorot_uniform':
        initializer = tf.glorot_uniform_initializer(seed)
    elif init_op == 'constant':
        initializer = tf.constant_initializer(bias_start)
    else:
        raise ValueError("Unknown init_op %s" % init_op)
    
    return initializer
    
def get_device_str(device_id, num_gpus):
    """Return a device string."""
    if num_gpus == 0:
        return "/cpu:0"
    device_str = "/gpu:%d" % (device_id % num_gpus)
    return device_str

def create_train_model(flags, model_creator):
    """Create train model and graph."""
    graph = tf.Graph()
    with graph.as_default():
        model = model_creator(flags, mode=tf.estimator.ModeKeys.TRAIN)
        return TrainModel(graph=graph, model=model)
    
def create_infer_model(flags, model_creator):
    """Create infer model and graph."""
    graph = tf.Graph()
    with graph.as_default():
        model = model_creator(flags, mode=tf.estimator.ModeKeys.PREDICT)
        return InferModel(graph=graph, model=model)
    
def create_or_load_model(model_dir, model, sess):
    """If there's a ckpt file, load the file.
       Otherwise, initialize variables."""
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model.saver.restore(sess, latest_ckpt)
    else:
        sess.run(tf.global_variables_initializer())

    global_step = model.global_step.eval(session=sess)
    return model, global_step
