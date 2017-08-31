import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import numpy as np

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

#----------------------------------------------------------
#  more NN helper function
#----------------------------------------------------------
# Mostly bilinear interpolation
def _variable_with_weight_decay(shape, stddev, wd, decoder=False):
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable('weights', shape=shape, initializer=initializer)

    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        if not decoder:
            tf.add_to_collection('losses', weight_decay)
        else:
            tf.add_to_collection('dec_losses', weight_decay)

    # TODO: add variable_summary
    return var

def get_deconv_filter(f_shape):
    width = f_shape[0]
    height = f_shape[1]
    # f = ceil(width/2.0)
    f = width//2.0
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    var = tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)
    return var

def _bias_variable(shape, constant=0.0):
    initializer = tf.constant_initializer(constant)
    var = tf.get_variable(name='biases', shape=shape, initializer=initializer)

    # TODO: variable summary
    return var

def _score_layer(bottom, name, num_classes):
    with tf.variable_scope(name) as scope:
        # get number of input channels
        in_features = bottom.get_shape()[3].value
        shape = [1, 1, in_features, num_classes]

        # What is stddev for?
        if name == 'score_fr':
            num_input = in_features
            stddev = (2/num_input)**0.5
        elif name == 'pool4':
            stddev = 1e-3
        elif name == 'pool3':
            stddev = 1e-4
        # Apply convolution

        # Hyperparameter
        w_decay = 5e-4

        weights = _variable_with_weight_decay(shape, stddev, w_decay, decoder=True)
        conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')

        conv_biases = _bias_variable([num_classes], constant=0.0)
        bias = tf.nn.bias_add(conv, conv_biases)
    return bias

def _upsample_layer(bottom, shape, num_classes, name, debug,
                    ksize=4, stride=2):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name) as scope:
        in_features = bottom.get_shape()[3].value

        if shape is None:
    #         compute shape out of bottom
            in_shape = tf.shape(bottom)
            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]

        output_shape = tf.stack(new_shape)

        # What would we get from observing stddev?
        f_shape = [ksize, ksize, num_classes, in_features]

        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')
        helper._activation_summary(deconv)
    return deconv

#----------------------------------------------------------
#  end of NN helper function
#----------------------------------------------------------

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    default_graph = tf.get_default_graph()
    vgg_input_tensor = default_graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = default_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = default_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tentor = default_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = default_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tentor, vgg_layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # upsampled2 = _upsample_layer(vgg_layer7_out, shape=tf.shape(vgg_layer4_out),
    #                              num_classes=num_classes, name="upsampled2", debug=False, ksize=4, stride=2)
    # fuse_pool4 = tf.add(upsampled2, vgg_layer4_out)
    #
    # upsampled4 = _upsample_layer(fuse_pool4, shape=tf.shape(vgg_layer3_out), num_classes=num_classes, name="upsampled4",
    #                              debug=False, ksize=16, stride=8)
    # fuse_pool3 = tf.add(upsampled4, vgg_layer3_out)
    # return tf.argmax(fuse_pool3, dimension=3)

#     Using udacity dimensions
    stddev = 0.01
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    vgg_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1, 1), kernel_initializer=initializer)

    upsample_vgg_layer7 = tf.layers.conv2d_transpose(vgg_layer7, num_classes, 4, strides=(2, 2))

    pool4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1, 1), kernel_initializer=initializer)
    fcn_layer1 = tf.add(upsample_vgg_layer7, pool4)

    # Add a 1x1 conv layer here?
    fcn_layer2 = tf.layers.conv2d_transpose(fcn_layer1, num_classes, 4, strides=(2, 2))

    vgg_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1, 1), kernel_initializer=initializer)
    combined_layer2 = tf.add(vgg_layer3, fcn_layer2)

    upscore32 = tf.layers.conv2d_transpose(combined_layer2, num_classes, 16, strides=(8, 8))


    # logits = tf.reshape(upscore32, (-1, num_classes))
    # return logits
    return upscore32

tests.test_layers(layers)


# def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
#     """
#     Build the TensorFLow loss and optimizer operations.
#     :param nn_last_layer: TF Tensor of the last layer in the neural network
#     :param correct_label: TF Placeholder for the correct label image
#     :param learning_rate: TF Placeholder for the learning rate
#     :param num_classes: Number of classes to classify
#     :return: Tuple of (logits, train_op, cross_entropy_loss)
#     """
#     # TODO: Implement function
#     return None, None, None
# tests.test_optimize(optimize)
#
#
# def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
#              correct_label, keep_prob, learning_rate):
#     """
#     Train neural network and print out the loss during training.
#     :param sess: TF Session
#     :param epochs: Number of epochs
#     :param batch_size: Batch size
#     :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
#     :param train_op: TF Operation to train the neural network
#     :param cross_entropy_loss: TF Tensor for the amount of loss
#     :param input_image: TF Placeholder for input images
#     :param correct_label: TF Placeholder for label images
#     :param keep_prob: TF Placeholder for dropout keep probability
#     :param learning_rate: TF Placeholder for learning rate
#     """
#     # TODO: Implement function
#     pass
# tests.test_train_nn(train_nn)





def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Hyperparameters
    batch_size = 6

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_t, keep_prob_t, layer3_out_t, layer4_out_t, layer7_out_t = load_vgg(sess, vgg_path);
        layers(layer3_out_t, layer4_out_t, layer7_out_t, num_classes)

        # Tensor summary
        # Seems for now the summary doesn't work
        # helper._activation_summary(input_t)
        # helper._activation_summary(keep_prob_t)

        # TODO: Train NN using the train_nn function
    #     Try loading the new data now
        gen = get_batches_fn(batch_size)
        for batch_x, batch_y in gen:
            print(batch_x, batch_y)

        # sess.run()
    #     # TODO: Save inference data using helper.save_inference_samples
    #     #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
    #
    #     # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
