import numpy as np
import tensorflow as tf

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
