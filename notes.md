Papers to read

[For understanding convolution](https://arxiv.org/pdf/1603.07285.pdf)

[The difference between convolution and cross correlation from Signal Analysis](https://dsp.stackexchange.com/questions/27451/the-difference-between-convolution-and-cross-correlation-from-a-signal-analysis)

[Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

[CS231n Winter 2016: Lecture 13: Segmentation, soft attention, spatial transformers](https://www.youtube.com/watch?v=ByjaPdWXKJ4)

Semantic Segmentation
Multi-Scale testing

Iterative refinement

I'm working on semantic segmentation project and following steps were used to build the model.

1. Download `vgg16_weights.npz` and initialize encoder layer from the pre-trained weights.
2. Next, create three fully-convolutional layers `[512, 4096]`, `[4096, 4096]`, `[4096, 2]`, initialized kernels with `tf.truncated_normal(.., stddev=0.02)`
3. Created three upsampling layers `[kernel=4, stride=2]`, `[kernel=4, stride=2]`, `[kernel=16, stride=8]` and initialized kernel with `tf.truncated_normal(..., stddev=0.02)`
4. Two skipped layer connections were used.
5. The network was trained with `tf.train.AdamOptimizer(1e-4)` with `100` epochs and I can see the training loss is decreasing.
However, when I test it against testing data set, the result was not visually appealing (attached two samples).

Obviously, I can increase the number of epochs, other than that what are some tricks I can use to increase the quality of the output?


Question

1. From the test cases it appears that layer3 has 256 channels, layer4 has 512 channels, and layer7 has 4096 channels. However looking at the VGG paper: https://arxiv.org/pdf/1409.1556.pdf, layer3 and 4 should have 128 channels and layer 7 should have 256 channels?
2. If layer7 has 4096 channels would that means it is the output of a fully-connected layer? There is no FC layer in FCN right?

Answer


ng et al. changed the original VGG-16, what they call conv7  is actually convolutionalized fc7 - see https://arxiv.org/pdf/1411.4038.pdf page 5.

https://discussions.udacity.com/t/what-is-the-output-layer-of-the-pre-trained-vgg16-to-be-fed-to-layers-project/327033/16

By looking at the pre-trained weights, the pretrained VGG16 has already had the 1x1 convolution layers
'

Using tensorboard:
 tensorboard --logdir=log/

 localhost:6006

Implementation references:

[Marvin Teichmann FCN8 VGG](https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn8_vgg.py)
