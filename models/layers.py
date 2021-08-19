'''
Neural network layers, originally from PointNet++ code.

Modified by Zi Jian to add pairwise_dist computation layer

Tf2 refactor attempt by TianYi
'''

import tensorflow as tf

@tf.function
def pairwise_dist(A: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
    ''' Computes pairwise distance
    :param A: (B x N x D) containing descriptors of A
    :param B: (B x N x D) containing descriptors of B
    :return: (B x N x N) tensor. Element[i,j,k] denotes the distance between the jth descriptor in ith model of A,
             and kth descriptor in ith model of B
    '''

    A = tf.expand_dims(A, axis=2)
    B = tf.expand_dims(B, axis=1)
    dist = tf.reduce_sum(tf.math.squared_difference(A, B), axis=3)

    return dist

class MaxPoolAxis(tf.keras.layers.Layer):
    '''Computes custom max pooling operation on custom axis.
    '''
    def __init__(self, name="MaxPoolAxis", axis:'list[int]'=[2]):
        super(MaxPoolAxis, self).__init__(name=name)
        self.axis = axis

    def call(self, input_pts, training):
        pooled = tf.reduce_max(input_pts, axis=self.axis, keepdims=True)

        return pooled

class MaxPoolConcat(tf.keras.layers.Layer):
    ''' Computes custom max pooling operation on custom axis,
        and then applies expansion and concatenation.
        Inherits from maxPoolAxis.
    '''
    def __init__(self, name="MaxPoolConcat", axis:'list[int]'=[2]):
        super(MaxPoolConcat, self).__init__(name=name)
        self.maxPoolAxis = MaxPoolAxis(axis=axis)

    def call(self, input_pts: tf.Variable, training):
        pooled = self.maxPoolAxis(input_pts)
        pooled_tile = tf.tile(pooled, [1, 1, input_pts.shape[2], 1])
        pooled_out = tf.concat((input_pts, pooled_tile), axis=3)

        return pooled_out

# Implements the conv2d with BN operation as found in feature_detection_module in the original code
# Perhaps losses need to be implemented here with the add_loss() method
class Conv2D_BN(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, bn=True, stride=[1,1], padding='same', 
                activation='relu', name='conv2d_bn'):
        super(Conv2D_BN, self).__init__(name=name)
        
        conv_name = name+"_conv2d"
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding, 
                                    activation=None if bn else activation,
                                    name=conv_name)

        self.bn = None
        self.activ = None
        
        if bn:
            bn_name = name+"_bn"
            self.bn = tf.keras.layers.BatchNormalization( axis=-1, name=bn_name)

            if activation is not None:
                activ_name = name+"_activ_"+activation
                self.activ = tf.keras.layers.Activation(activation, name=activ_name)

    def call(self, inp:tf.Variable, training=False):
        conv_out = self.conv(inp, training=training)
        # self.input_shape = self.conv.input_shape
        # self.output_shape = self.conv.output_shape

        if self.bn is not None:
            bn_out = self.bn(conv_out, training=training)
            # self.output_shape = self.bn.output_shape
        if self.activ is not None:
            bn_out = self.activ(bn_out, training=training)
            # self.output_shape = self.activ.output_shape
        
        return bn_out

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if use_fp16 else tf.float32
    # var = tf.compat.v1.get_variable(name, shape, initializer=initializer, dtype=dtype)
    var = tf.Variable(initial_value=initializer, name=name, shape=shape, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
      use_xavier: bool, whether to use xavier initializer

    Returns:
      Variable Tensor
    """
    if use_xavier:
        # initializer = tf.compat.v1.keras.initializers.glorot_normal()
        initializer = tf.keras.initializers.GlorotNormal()
    else:
        # initializer = tf.compat.v1.truncated_normal_initializer(stddev=stddev)
        initializer = tf.keras.initializers.TruncatedNormal(stddev=stddev)
    
    var = _variable_on_cpu(name, shape, initializer)

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.compat.v1.add_to_collection('losses', weight_decay)   # This needs to be changed 
    return var


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
    """ Dropout layer.

    Args:
      inputs: tensor
      is_training: boolean tf.compat.v1.Variable
      scope: string
      keep_prob: float in [0,1]
      noise_shape: list of ints

    Returns:
      tensor variable
    """
    with tf.compat.v1.variable_scope(scope) as sc:
        outputs = tf.cond(is_training,
                          lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                          lambda: inputs)
        return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """
    with tf.compat.v1.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1]
        weights = _variable_with_weight_decay('weights',
                                              shape=[num_input_units, num_outputs],
                                              use_xavier=use_xavier,
                                              stddev=stddev,
                                              wd=weight_decay)
        outputs = tf.matmul(inputs, weights)
        biases = _variable_on_cpu('biases', [num_outputs],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
    """ Batch normalization on FC data.

    Args:
        inputs:      Tensor, 2D BxC input
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, ], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 1D convolutional maps.

    Args:
        inputs:      Tensor, 3D BLC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1], bn_decay)


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 2D convolutional maps.

    Args:
        inputs:      Tensor, 4D BHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2], bn_decay)


def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 3D convolutional maps.

    Args:
        inputs:      Tensor, 5D BDHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2, 3], bn_decay)

def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow.

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    # For support of GAN
    # bn_decay = bn_decay if bn_decay is not None else 0.9
    # return tf.contrib.layers.batch_norm(inputs,
    #                                    center=True, scale=True,
    #                                    is_training=is_training, decay=bn_decay,updates_collections=None,
    #                                    scope=scope)
    with tf.compat.v1.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1]
        beta = tf.compat.v1.Variable(tf.constant(0.0, shape=[num_channels]),
                           name='beta', trainable=True)
        gamma = tf.compat.v1.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        # Need to set reuse=False, otherwise if reuse, will see moments_1/mean/ExponentialMovingAverage/ does not exist
        # https://github.com/shekkizh/WassersteinGAN.tensorflow/issues/3
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=False):
            ema_apply_op = tf.cond(is_training,
                                   lambda: ema.apply([batch_mean, batch_var]),
                                   lambda: tf.no_op())

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed