import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl

DEFAULT_PADDING = 'VALID'
DEFAULT_DATAFORMAT = 'NHWC'

BN_param_map = {'scale': 'gamma',
                'offset': 'beta',
                'variance': 'moving_variance',
                'mean': 'moving_mean'}


# 装饰器， 构建每层网络的时候添加操作
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # 提取操作的名字，如果没有，设置为 name
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # 获得网络的输入
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # 执行装饰的操作
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # 这一层的输出是下一层的输入
        self.feed(layer_output)
        # 返回self支持链式
        return self

    return layer_decorated

# 网络最后的输出是输入的 1 / 8
class Network(object):

    def __init__(self, conf):
        # The input nodes for this network
        self.conf = conf
        # The current list of terminal nodes
        self.terminals = []
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0), shape=[], name='use_dropout')
        pass

    # 运行网络, 返回logits
    def fit(self, inputs):
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # 构建网络
        self.setup(self.conf.num_classes)
        return self.layers['conv6']

    # 网络架构
    def setup(self, num_classes):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')
        pass

        # 加载网络权重，从np文件中

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path, encoding='latin1').items()

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].items():
                    try:
                        if 'bn' in op_name:
                            param_name = BN_param_map[param_name]
                            data = np.squeeze(data)

                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

                            # 设置本层的输出为下一层的输入

    def feed(self, *args):
        assert len(args) != 0
        # 清空下一层的输入队列
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

        # 获得网络最后的输出

    def get_output(self):
        return self.terminals[-1]

        # 返回一个 index-前缀的唯一的名字，

    def get_unique_name(self, prefix):
        return '{}_{}'.format(prefix, sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1)

    # 创建新的tensor
    def make_var(self, name, shape):
        return tf.get_variable(name, shape, initializer=tcl.xavier_initializer())

    # 获得层的名称
    def get_layer_name(self):
        return self.layers.keys()

    # 判断给定的padding方式是不是合法的
    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def zero_padding(self, input, paddings, name):
        pad_mat = np.array([[0, 0], [paddings, paddings], [paddings, paddings], [0, 0]])
        return tf.pad(input, paddings=pad_mat, name=name)

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, biased=True):
        # 检验padding方式
        self.validate_padding(padding)
        # channel
        c_i = input.get_shape()[-1]
        # 堆叠卷积
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i, c_o])
            output = tf.nn.conv2d(input, kernel, [1, s_h, s_w, 1], padding=padding, data_format=DEFAULT_DATAFORMAT)

            if biased:
                output = tf.nn.bias_add(output, self.make_var('biases', [c_o]))
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def atrous_conv(self, input, k_h, k_w, c_o, dilation, name, relu=True, padding=DEFAULT_PADDING, biased=True):
        # 验证padding‘方式
        self.validate_padding(padding)
        # channel
        c_i = input.get_shape()[-1]

        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i, c_o])
            output = tf.nn.atrous_conv2d(input, kernel, dilation, padding=padding)

            if biased:
                output = tf.nn.bias_add(output, self.make_var('biases', [c_o]))
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1],
                              padding=padding, name=name, data_format=DEFAULT_DATAFORMAT)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1],
                              padding=padding, name=name, data_format=DEFAULT_DATAFORMAT)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # 输入的维度
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            return op(feed_in, weights, biases, name=scope.name)

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                return tf.nn.softmax(input, name)

    @layer
    def batch_normalization(self, input, name, relu=False):
        output = tf.layers.batch_normalization(input, training=self.conf.is_training, name=name)
        if relu:
            output = tf.nn.relu(output)
        return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)

    @layer
    def resize_bilinear(self, input, size, name):
        return tf.image.resize_bilinear(input, size=size, align_corners=True, name=name)
    pass


class PSPNet101(Network):
    # 网络架构
    def setup(self, num_classes):

        (self.feed('data')  # 720x720x3
         .conv(3, 3, 64, 2, 2, biased=False, relu=False, padding='SAME', name='conv1_1_3x3_s2')  # 360x360x64
         .batch_normalization(relu=False, name='conv1_1_3x3_s2_bn')
         .relu(name='conv1_1_3x3_s2_bn_relu')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_2_3x3')  # 360x360x64
         .batch_normalization(relu=True, name='conv1_2_3x3_bn')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_3_3x3')  # 360x360x128
         .batch_normalization(relu=True, name='conv1_3_3x3_bn')
         .max_pool(3, 3, 2, 2, padding='SAME', name='pool1_3x3_s2')  # 180x180x128
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_1_1x1_proj')  # 180x180x256
         .batch_normalization(relu=False, name='conv2_1_1x1_proj_bn'))

        (self.feed('pool1_3x3_s2')  # 180x180x128
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_1_1x1_reduce')  # 180x180x64
         .batch_normalization(relu=True, name='conv2_1_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding1')  # 182x182x64
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_1_3x3')  # 180x180x64
         .batch_normalization(relu=True, name='conv2_1_3x3_bn')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_1_1x1_increase')  # 180x180x256
         .batch_normalization(relu=False, name='conv2_1_1x1_increase_bn'))

        (self.feed('conv2_1_1x1_proj_bn',
                   'conv2_1_1x1_increase_bn')
         .add(name='conv2_1')  # 180x180x256
         .relu(name='conv2_1/relu')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_2_1x1_reduce')  # 180x180x64
         .batch_normalization(relu=True, name='conv2_2_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding2')  # 182x182x64
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_2_3x3')  # 180x180x64
         .batch_normalization(relu=True, name='conv2_2_3x3_bn')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_2_1x1_increase')  # 180x180x256
         .batch_normalization(relu=False, name='conv2_2_1x1_increase_bn'))

        (self.feed('conv2_1/relu',
                   'conv2_2_1x1_increase_bn')
         .add(name='conv2_2')
         .relu(name='conv2_2/relu')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_3_1x1_reduce')  # 180x180x64
         .batch_normalization(relu=True, name='conv2_3_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding3')  # 182x182x64
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_3_3x3')  # 180x180x64
         .batch_normalization(relu=True, name='conv2_3_3x3_bn')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_3_1x1_increase')  # 180x180x256
         .batch_normalization(relu=False, name='conv2_3_1x1_increase_bn'))

        (self.feed('conv2_2/relu',
                   'conv2_3_1x1_increase_bn')
         .add(name='conv2_3')
         .relu(name='conv2_3/relu')  # 180x180x256
         .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='conv3_1_1x1_proj')  # 90x90x512
         .batch_normalization(relu=False, name='conv3_1_1x1_proj_bn'))

        (self.feed('conv2_3/relu')  # 180x180x256
         .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='conv3_1_1x1_reduce')  # 90x90x128
         .batch_normalization(relu=True, name='conv3_1_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding4')  # 92x92x128
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_1_3x3')  # 90x90x128
         .batch_normalization(relu=True, name='conv3_1_3x3_bn')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_1_1x1_increase')  # 90x90x512
         .batch_normalization(relu=False, name='conv3_1_1x1_increase_bn'))

        (self.feed('conv3_1_1x1_proj_bn',
                   'conv3_1_1x1_increase_bn')
         .add(name='conv3_1')  # 90x90x512
         .relu(name='conv3_1/relu')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_2_1x1_reduce')  # 90x90x128
         .batch_normalization(relu=True, name='conv3_2_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding5')  # 92x92x512
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_2_3x3')  # 90x90x128
         .batch_normalization(relu=True, name='conv3_2_3x3_bn')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_2_1x1_increase')  # 90x90x512
         .batch_normalization(relu=False, name='conv3_2_1x1_increase_bn'))

        (self.feed('conv3_1/relu',
                   'conv3_2_1x1_increase_bn')  # 90x90x512
         .add(name='conv3_2')
         .relu(name='conv3_2/relu')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_3_1x1_reduce')  # 90x90x128
         .batch_normalization(relu=True, name='conv3_3_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding6')  # 92x92x128
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_3_3x3')  # 90x90x128
         .batch_normalization(relu=True, name='conv3_3_3x3_bn')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_3_1x1_increase')  # 90x90x512
         .batch_normalization(relu=False, name='conv3_3_1x1_increase_bn'))

        (self.feed('conv3_2/relu',
                   'conv3_3_1x1_increase_bn')
         .add(name='conv3_3')  # 90x90x512
         .relu(name='conv3_3/relu')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_4_1x1_reduce')  # 90x90x128
         .batch_normalization(relu=True, name='conv3_4_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding7')  # 92x92x512
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_4_3x3')  # 90x90x128
         .batch_normalization(relu=True, name='conv3_4_3x3_bn')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_4_1x1_increase')  # 90x90x512
         .batch_normalization(relu=False, name='conv3_4_1x1_increase_bn'))

        (self.feed('conv3_3/relu',
                   'conv3_4_1x1_increase_bn')
         .add(name='conv3_4')  # 90x90x512
         .relu(name='conv3_4/relu')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_1x1_proj')  # 90x90x1024
         .batch_normalization(relu=False, name='conv4_1_1x1_proj_bn'))

        (self.feed('conv3_4/relu')  # 90x90x512
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_1_1x1_reduce')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_1_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding8')  # 92x92x256
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_1_3x3')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_1_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_1x1_increase')  # 90x90x1024
         .batch_normalization(relu=False, name='conv4_1_1x1_increase_bn'))

        (self.feed('conv4_1_1x1_proj_bn',
                   'conv4_1_1x1_increase_bn')
         .add(name='conv4_1')  # 90x90x1024
         .relu(name='conv4_1/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_2_1x1_reduce')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_2_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding9')  # 92x92x256
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_2_3x3')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_2_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_2_1x1_increase')  # 90x90x1024
         .batch_normalization(relu=False, name='conv4_2_1x1_increase_bn'))

        (self.feed('conv4_1/relu',
                   'conv4_2_1x1_increase_bn')
         .add(name='conv4_2')  # 90x90x1024
         .relu(name='conv4_2/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_3_1x1_reduce')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_3_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding10')  # 92x92x256
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_3_3x3')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_3_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_3_1x1_increase')  # 90x90x1024
         .batch_normalization(relu=False, name='conv4_3_1x1_increase_bn'))

        (self.feed('conv4_2/relu',
                   'conv4_3_1x1_increase_bn')
         .add(name='conv4_3')  # 90x90x1024
         .relu(name='conv4_3/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_4_1x1_reduce')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_4_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding11')  # 92x92x256
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_4_3x3')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_4_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_4_1x1_increase')  # 90x90x1024
         .batch_normalization(relu=False, name='conv4_4_1x1_increase_bn'))

        (self.feed('conv4_3/relu',
                   'conv4_4_1x1_increase_bn')  # 90x90x1024
         .add(name='conv4_4')
         .relu(name='conv4_4/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_5_1x1_reduce')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_5_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding12')  # 92x92x256
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_5_3x3')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_5_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_5_1x1_increase')  # 90x90x1024
         .batch_normalization(relu=False, name='conv4_5_1x1_increase_bn'))

        (self.feed('conv4_4/relu',
                   'conv4_5_1x1_increase_bn')
         .add(name='conv4_5')
         .relu(name='conv4_5/relu')  # 90x90x1024
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_6_1x1_reduce')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_6_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding13')  # 92x92x256
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_6_3x3')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_6_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_6_1x1_increase')  # 90x90x1024
         .batch_normalization(relu=False, name='conv4_6_1x1_increase_bn'))

        (self.feed('conv4_5/relu',
                   'conv4_6_1x1_increase_bn')  # 90x90x1024
         .add(name='conv4_6')
         .relu(name='conv4_6/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_7_1x1_reduce')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_7_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding14')  # 92x92x256
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_7_3x3')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_7_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_7_1x1_increase')  # 90x90x1024
         .batch_normalization(relu=False, name='conv4_7_1x1_increase_bn'))

        (self.feed('conv4_6/relu',
                   'conv4_7_1x1_increase_bn')  # 90x90x1024
         .add(name='conv4_7')
         .relu(name='conv4_7/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_8_1x1_reduce')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_8_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding15')  # 92x92x256
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_8_3x3')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_8_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_8_1x1_increase')  # 90x90x1024
         .batch_normalization(relu=False, name='conv4_8_1x1_increase_bn'))

        (self.feed('conv4_7/relu',
                   'conv4_8_1x1_increase_bn')
         .add(name='conv4_8')
         .relu(name='conv4_8/relu')  # 90x90x1024
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_9_1x1_reduce')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_9_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding16')
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_9_3x3')  # 90x90x256
         .batch_normalization(relu=True, name='conv4_9_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_9_1x1_increase')  # 90x90x1024
         .batch_normalization(relu=False, name='conv4_9_1x1_increase_bn'))

        (self.feed('conv4_8/relu',
                   'conv4_9_1x1_increase_bn')
         .add(name='conv4_9')
         .relu(name='conv4_9/relu')  # 90x90x1024
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_10_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_10_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding17')
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_10_3x3')
         .batch_normalization(relu=True, name='conv4_10_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_10_1x1_increase')  # 90x90x1024
         .batch_normalization(relu=False, name='conv4_10_1x1_increase_bn'))

        (self.feed('conv4_9/relu',
                   'conv4_10_1x1_increase_bn')
         .add(name='conv4_10')  # 90x90x1024
         .relu(name='conv4_10/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_11_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_11_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding18')
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_11_3x3')
         .batch_normalization(relu=True, name='conv4_11_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_11_1x1_increase')  # 90x90x1024
         .batch_normalization(relu=False, name='conv4_11_1x1_increase_bn'))

        (self.feed('conv4_10/relu',
                   'conv4_11_1x1_increase_bn')
         .add(name='conv4_11')
         .relu(name='conv4_11/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_12_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_12_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding19')
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_12_3x3')
         .batch_normalization(relu=True, name='conv4_12_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_12_1x1_increase')  # 90x90x1024
         .batch_normalization(relu=False, name='conv4_12_1x1_increase_bn'))

        (self.feed('conv4_11/relu',
                   'conv4_12_1x1_increase_bn')
         .add(name='conv4_12')
         .relu(name='conv4_12/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_13_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_13_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding20')
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_13_3x3')
         .batch_normalization(relu=True, name='conv4_13_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_13_1x1_increase')
         .batch_normalization(relu=False, name='conv4_13_1x1_increase_bn'))  # 90x90x1024

        (self.feed('conv4_12/relu',
                   'conv4_13_1x1_increase_bn')
         .add(name='conv4_13')
         .relu(name='conv4_13/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_14_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_14_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding21')
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_14_3x3')
         .batch_normalization(relu=True, name='conv4_14_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_14_1x1_increase')
         .batch_normalization(relu=False, name='conv4_14_1x1_increase_bn'))  # 90x90x1024

        (self.feed('conv4_13/relu',
                   'conv4_14_1x1_increase_bn')
         .add(name='conv4_14')
         .relu(name='conv4_14/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_15_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_15_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding22')
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_15_3x3')
         .batch_normalization(relu=True, name='conv4_15_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_15_1x1_increase')
         .batch_normalization(relu=False, name='conv4_15_1x1_increase_bn'))  # 90x90x1024

        (self.feed('conv4_14/relu',
                   'conv4_15_1x1_increase_bn')
         .add(name='conv4_15')
         .relu(name='conv4_15/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_16_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_16_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding23')
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_16_3x3')
         .batch_normalization(relu=True, name='conv4_16_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_16_1x1_increase')
         .batch_normalization(relu=False, name='conv4_16_1x1_increase_bn'))  # 90x90x1024

        (self.feed('conv4_15/relu',
                   'conv4_16_1x1_increase_bn')
         .add(name='conv4_16')
         .relu(name='conv4_16/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_17_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_17_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding24')
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_17_3x3')
         .batch_normalization(relu=True, name='conv4_17_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_17_1x1_increase')
         .batch_normalization(relu=False, name='conv4_17_1x1_increase_bn'))  # 90x90x1024

        (self.feed('conv4_16/relu',
                   'conv4_17_1x1_increase_bn')
         .add(name='conv4_17')
         .relu(name='conv4_17/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_18_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_18_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding25')
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_18_3x3')
         .batch_normalization(relu=True, name='conv4_18_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_18_1x1_increase')
         .batch_normalization(relu=False, name='conv4_18_1x1_increase_bn'))  # 90x90x1024

        (self.feed('conv4_17/relu',
                   'conv4_18_1x1_increase_bn')
         .add(name='conv4_18')
         .relu(name='conv4_18/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_19_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_19_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding26')
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_19_3x3')
         .batch_normalization(relu=True, name='conv4_19_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_19_1x1_increase')
         .batch_normalization(relu=False, name='conv4_19_1x1_increase_bn'))  # 90x90x1024

        (self.feed('conv4_18/relu',
                   'conv4_19_1x1_increase_bn')
         .add(name='conv4_19')
         .relu(name='conv4_19/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_20_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_20_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding27')
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_20_3x3')
         .batch_normalization(relu=True, name='conv4_20_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_20_1x1_increase')
         .batch_normalization(relu=False, name='conv4_20_1x1_increase_bn'))  # 90x90x1024

        (self.feed('conv4_19/relu',
                   'conv4_20_1x1_increase_bn')
         .add(name='conv4_20')
         .relu(name='conv4_20/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_21_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_21_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding28')
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_21_3x3')
         .batch_normalization(relu=True, name='conv4_21_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_21_1x1_increase')
         .batch_normalization(relu=False, name='conv4_21_1x1_increase_bn'))  # 90x90x1024

        (self.feed('conv4_20/relu',
                   'conv4_21_1x1_increase_bn')
         .add(name='conv4_21')
         .relu(name='conv4_21/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_22_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_22_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding29')
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_22_3x3')
         .batch_normalization(relu=True, name='conv4_22_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_22_1x1_increase')
         .batch_normalization(relu=False, name='conv4_22_1x1_increase_bn'))  # 90x90x1024

        (self.feed('conv4_21/relu',
                   'conv4_22_1x1_increase_bn')
         .add(name='conv4_22')
         .relu(name='conv4_22/relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_23_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_23_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding30')
         .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_23_3x3')
         .batch_normalization(relu=True, name='conv4_23_3x3_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_23_1x1_increase')
         .batch_normalization(relu=False, name='conv4_23_1x1_increase_bn'))  # 90x90x1024

        (self.feed('conv4_22/relu',
                   'conv4_23_1x1_increase_bn')
         .add(name='conv4_23')
         .relu(name='conv4_23/relu')
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_1_1x1_proj')  # 90x90x2048
         .batch_normalization(relu=False, name='conv5_1_1x1_proj_bn'))

        (self.feed('conv4_23/relu')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_1_1x1_reduce')
         .batch_normalization(relu=True, name='conv5_1_1x1_reduce_bn')
         .zero_padding(paddings=4, name='padding31')
         .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_1_3x3')
         .batch_normalization(relu=True, name='conv5_1_3x3_bn')
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_1_1x1_increase')
         .batch_normalization(relu=False, name='conv5_1_1x1_increase_bn'))  # 90x90x2048

        (self.feed('conv5_1_1x1_proj_bn',
                   'conv5_1_1x1_increase_bn')
         .add(name='conv5_1')
         .relu(name='conv5_1/relu')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_2_1x1_reduce')
         .batch_normalization(relu=True, name='conv5_2_1x1_reduce_bn')
         .zero_padding(paddings=4, name='padding32')
         .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_2_3x3')
         .batch_normalization(relu=True, name='conv5_2_3x3_bn')
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_2_1x1_increase')
         .batch_normalization(relu=False, name='conv5_2_1x1_increase_bn'))  # 90x90x2048

        (self.feed('conv5_1/relu',
                   'conv5_2_1x1_increase_bn')
         .add(name='conv5_2')
         .relu(name='conv5_2/relu')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_1x1_reduce')
         .batch_normalization(relu=True, name='conv5_3_1x1_reduce_bn')
         .zero_padding(paddings=4, name='padding33')
         .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_3_3x3')
         .batch_normalization(relu=True, name='conv5_3_3x3_bn')
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_3_1x1_increase')
         .batch_normalization(relu=False, name='conv5_3_1x1_increase_bn'))  # 90x90x2048

        (self.feed('conv5_2/relu',
                   'conv5_3_1x1_increase_bn')
         .add(name='conv5_3')
         .relu(name='conv5_3/relu'))  # 90x90x2048

        conv5_3 = self.layers['conv5_3/relu']
        shape = tf.shape(conv5_3)[1:3]

        output = 10
        (self.feed('conv5_3/relu')
         .avg_pool(output * 6, output * 6, output * 6, output * 6, name='conv5_3_pool1')  # 1x1x2048
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool1_conv')
         .batch_normalization(relu=True, name='conv5_3_pool1_conv_bn')
         .resize_bilinear(shape, name='conv5_3_pool1_interp'))  # 90x90x512

        (self.feed('conv5_3/relu')
         .avg_pool(output * 3, output * 3, output * 3, output * 3, name='conv5_3_pool2')  # 2x2x2048
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool2_conv')
         .batch_normalization(relu=True, name='conv5_3_pool2_conv_bn')
         .resize_bilinear(shape, name='conv5_3_pool2_interp'))  # 90x90x512

        (self.feed('conv5_3/relu')
         .avg_pool(output * 2, output * 2, output * 2, output * 2, name='conv5_3_pool3')  # 3x3x2048
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool3_conv')
         .batch_normalization(relu=True, name='conv5_3_pool3_conv_bn')
         .resize_bilinear(shape, name='conv5_3_pool3_interp'))  # 90x90x512

        (self.feed('conv5_3/relu')
         .avg_pool(output, output, output, output, name='conv5_3_pool6')  # 4x4x2048
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool6_conv')
         .batch_normalization(relu=True, name='conv5_3_pool6_conv_bn')
         .resize_bilinear(shape, name='conv5_3_pool6_interp'))  # 90x90x512

        (self.feed('conv5_3/relu',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
         .concat(axis=-1, name='conv5_3_concat')  # 90x90x4096
         .conv(3, 3, 512, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_4')  # 90x90x512
         .batch_normalization(relu=True, name='conv5_4_bn')
         .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='conv6'))  # 90x90x21
        pass
    pass



class PSPNet50(Network):
    def setup(self, num_classes):
        '''Network definition.

        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
        '''
        (self.feed('data')
             .conv(3, 3, 64, 2, 2, biased=False, relu=False, padding='SAME', name='conv1_1_3x3_s2')
             .batch_normalization(relu=False, name='conv1_1_3x3_s2_bn')
             .relu(name='conv1_1_3x3_s2_bn_relu')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_2_3x3')
             .batch_normalization(relu=True, name='conv1_2_3x3_bn')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_3_3x3')
             .batch_normalization(relu=True, name='conv1_3_3x3_bn')
             .max_pool(3, 3, 2, 2, padding='SAME', name='pool1_3x3_s2')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_1_1x1_proj')
             .batch_normalization(relu=False, name='conv2_1_1x1_proj_bn'))

        (self.feed('pool1_3x3_s2')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_1_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding1')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_1_3x3')
             .batch_normalization(relu=True, name='conv2_1_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_1_1x1_increase')
             .batch_normalization(relu=False, name='conv2_1_1x1_increase_bn'))

        (self.feed('conv2_1_1x1_proj_bn',
                   'conv2_1_1x1_increase_bn')
             .add(name='conv2_1')
             .relu(name='conv2_1/relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_2_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding2')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_2_3x3')
             .batch_normalization(relu=True, name='conv2_2_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_2_1x1_increase')
             .batch_normalization(relu=False, name='conv2_2_1x1_increase_bn'))

        (self.feed('conv2_1/relu',
                   'conv2_2_1x1_increase_bn')
             .add(name='conv2_2')
             .relu(name='conv2_2/relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding3')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_3_3x3')
             .batch_normalization(relu=True, name='conv2_3_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_3_1x1_increase')
             .batch_normalization(relu=False, name='conv2_3_1x1_increase_bn'))

        (self.feed('conv2_2/relu',
                   'conv2_3_1x1_increase_bn')
             .add(name='conv2_3')
             .relu(name='conv2_3/relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='conv3_1_1x1_proj')
             .batch_normalization(relu=False, name='conv3_1_1x1_proj_bn'))

        (self.feed('conv2_3/relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='conv3_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_1_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding4')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_1_3x3')
             .batch_normalization(relu=True, name='conv3_1_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_1_1x1_increase')
             .batch_normalization(relu=False, name='conv3_1_1x1_increase_bn'))

        (self.feed('conv3_1_1x1_proj_bn',
                   'conv3_1_1x1_increase_bn')
             .add(name='conv3_1')
             .relu(name='conv3_1/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_2_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding5')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_2_3x3')
             .batch_normalization(relu=True, name='conv3_2_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_2_1x1_increase')
             .batch_normalization(relu=False, name='conv3_2_1x1_increase_bn'))

        (self.feed('conv3_1/relu',
                   'conv3_2_1x1_increase_bn')
             .add(name='conv3_2')
             .relu(name='conv3_2/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding6')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_3_3x3')
             .batch_normalization(relu=True, name='conv3_3_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_3_1x1_increase')
             .batch_normalization(relu=False, name='conv3_3_1x1_increase_bn'))

        (self.feed('conv3_2/relu',
                   'conv3_3_1x1_increase_bn')
             .add(name='conv3_3')
             .relu(name='conv3_3/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_4_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_4_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding7')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_4_3x3')
             .batch_normalization(relu=True, name='conv3_4_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_4_1x1_increase')
             .batch_normalization(relu=False, name='conv3_4_1x1_increase_bn'))

        (self.feed('conv3_3/relu',
                   'conv3_4_1x1_increase_bn')
             .add(name='conv3_4')
             .relu(name='conv3_4/relu')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_1x1_proj')
             .batch_normalization(relu=False, name='conv4_1_1x1_proj_bn'))

        (self.feed('conv3_4/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_1_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding8')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_1_3x3')
             .batch_normalization(relu=True, name='conv4_1_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_1x1_increase')
             .batch_normalization(relu=False, name='conv4_1_1x1_increase_bn'))

        (self.feed('conv4_1_1x1_proj_bn',
                   'conv4_1_1x1_increase_bn')
             .add(name='conv4_1')
             .relu(name='conv4_1/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_2_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding9')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_2_3x3')
             .batch_normalization(relu=True, name='conv4_2_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_2_1x1_increase')
             .batch_normalization(relu=False, name='conv4_2_1x1_increase_bn'))

        (self.feed('conv4_1/relu',
                   'conv4_2_1x1_increase_bn')
             .add(name='conv4_2')
             .relu(name='conv4_2/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_3_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding10')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_3_3x3')
             .batch_normalization(relu=True, name='conv4_3_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_3_1x1_increase')
             .batch_normalization(relu=False, name='conv4_3_1x1_increase_bn'))

        (self.feed('conv4_2/relu',
                   'conv4_3_1x1_increase_bn')
             .add(name='conv4_3')
             .relu(name='conv4_3/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_4_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_4_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding11')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_4_3x3')
             .batch_normalization(relu=True, name='conv4_4_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_4_1x1_increase')
             .batch_normalization(relu=False, name='conv4_4_1x1_increase_bn'))

        (self.feed('conv4_3/relu',
                   'conv4_4_1x1_increase_bn')
             .add(name='conv4_4')
             .relu(name='conv4_4/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_5_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_5_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding12')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_5_3x3')
             .batch_normalization(relu=True, name='conv4_5_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_5_1x1_increase')
             .batch_normalization(relu=False, name='conv4_5_1x1_increase_bn'))

        (self.feed('conv4_4/relu',
                   'conv4_5_1x1_increase_bn')
             .add(name='conv4_5')
             .relu(name='conv4_5/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_6_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_6_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding13')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_6_3x3')
             .batch_normalization(relu=True, name='conv4_6_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_6_1x1_increase')
             .batch_normalization(relu=False, name='conv4_6_1x1_increase_bn'))

        (self.feed('conv4_5/relu',
                   'conv4_6_1x1_increase_bn')
             .add(name='conv4_6')
             .relu(name='conv4_6/relu')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_1_1x1_proj')
             .batch_normalization(relu=False, name='conv5_1_1x1_proj_bn'))

        (self.feed('conv4_6/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_1_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding31')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_1_3x3')
             .batch_normalization(relu=True, name='conv5_1_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_1_1x1_increase')
             .batch_normalization(relu=False, name='conv5_1_1x1_increase_bn'))

        (self.feed('conv5_1_1x1_proj_bn',
                   'conv5_1_1x1_increase_bn')
             .add(name='conv5_1')
             .relu(name='conv5_1/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_2_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding32')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_2_3x3')
             .batch_normalization(relu=True, name='conv5_2_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_2_1x1_increase')
             .batch_normalization(relu=False, name='conv5_2_1x1_increase_bn'))

        (self.feed('conv5_1/relu',
                   'conv5_2_1x1_increase_bn')
             .add(name='conv5_2')
             .relu(name='conv5_2/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_3_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding33')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_3_3x3')
             .batch_normalization(relu=True, name='conv5_3_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_3_1x1_increase')
             .batch_normalization(relu=False, name='conv5_3_1x1_increase_bn'))

        (self.feed('conv5_2/relu',
                   'conv5_3_1x1_increase_bn')
             .add(name='conv5_3')
             .relu(name='conv5_3/relu'))

        conv5_3 = self.layers['conv5_3/relu']
        shape = tf.shape(conv5_3)[1:3]

        (self.feed('conv5_3/relu')
             .avg_pool(60, 60, 60, 60, name='conv5_3_pool1')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool1_conv')
             .batch_normalization(relu=True, name='conv5_3_pool1_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool1_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(30, 30, 30, 30, name='conv5_3_pool2')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool2_conv')
             .batch_normalization(relu=True, name='conv5_3_pool2_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool2_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(20, 20, 20, 20, name='conv5_3_pool3')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool3_conv')
             .batch_normalization(relu=True, name='conv5_3_pool3_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool3_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(10, 10, 10, 10, name='conv5_3_pool6')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool6_conv')
             .batch_normalization(relu=True, name='conv5_3_pool6_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool6_interp'))

        (self.feed('conv5_3/relu',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
             .concat(axis=-1, name='conv5_3_concat')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_4')
             .batch_normalization(relu=True, name='conv5_4_bn')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='conv6'))
        pass
    pass
