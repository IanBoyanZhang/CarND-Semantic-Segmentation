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
    l2_regularizier = tf.contrib.layers.l2_regularizer(1e-3)
    vgg_layer7_conv1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1, 1),
                                          kernel_initializer=initializer, padding='same',
                                          kernel_regularizer=l2_regularizier)

    output = tf.layers.conv2d_transpose(vgg_layer7_conv1x1, num_classes, 4, strides=(2, 2),
                                        padding='same', kernel_regularizer=l2_regularizier)

    pool4_conv1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1, 1),
                                     kernel_initializer=initializer, padding='same',
                                     kernel_regularizer=l2_regularizier)

    fcn_layer1 = tf.add(output, pool4_conv1x1)
    fcn_layer2 = tf.layers.conv2d_transpose(fcn_layer1, num_classes, 4, strides=(2, 2),
                                            padding='same', kernel_regularizer=l2_regularizier)

    vgg_layer3_conv1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1, 1),
                                          kernel_initializer=initializer, padding='same',
                                          kernel_regularizer=l2_regularizier)

    combined_layer2 = tf.add(vgg_layer3_conv1x1, fcn_layer2)
    fcn8 = tf.layers.conv2d_transpose(combined_layer2, num_classes, 16, strides=(8, 8),
                                      padding='same', kernel_regularizer=l2_regularizier)


#     pool4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1, 1), kernel_initializer=initializer)
#     fcn_layer1 = tf.add(upsample_vgg_layer7, pool4)
#
#     # Add a 1x1 conv layer here?
#     fcn_layer2 = tf.layers.conv2d_transpose(fcn_layer1, num_classes, 4, strides=(2, 2))
#
#     vgg_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1, 1), kernel_initializer=initializer)
#     combined_layer2 = tf.add(vgg_layer3, fcn_layer2)
#
#     upscore32 = tf.layers.conv2d_transpose(combined_layer2, num_classes, 16, strides=(8, 8))
#     Official layers
    return fcn8

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                                labels=correct_label))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss,
                                                              global_step=tf.train.get_global_step())
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())
    # https://discussions.udacity.com/t/implementation-of-train-nn/347885/5
    # keep_prob and learning rate hack
    for epoch in range(epochs):
        for images, labels in get_batches_fn(batch_size):
            feed = {
                input_image: images,
                correct_label: labels,
                keep_prob: 0.5,
                learning_rate: 0.001
            }
            _, out = sess.run([train_op, cross_entropy_loss], feed_dict=feed)
    pass
tests.test_train_nn(train_nn)

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
    learning_rate = 1e-5
    epochs = 10
    batch_size = 8
    keep_prob = 0.5

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_t, keep_prob_t, layer3_out_t, layer4_out_t, layer7_out_t = load_vgg(sess, vgg_path);
        nn_last_layer = layers(layer3_out_t, layer4_out_t, layer7_out_t, num_classes)

        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, num_classes, num_classes)


        # Tensor summary
        # TODO: Train NN using the train_nn function
    #     Try loading the new data now
    #     gen = get_batches_fn(batch_size)
    #     for batch_x, batch_y in gen:
    #         print(batch_x, batch_y)

        # sess.run()
    #     # TODO: Save inference data using helper.save_inference_samples
    #     #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
    #
    #     # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
