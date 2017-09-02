Papers to read

[For understanding convolution](https://arxiv.org/pdf/1603.07285.pdf)

[The difference between convolution and cross correlation from Signal Analysis](https://dsp.stackexchange.com/questions/27451/the-difference-between-convolution-and-cross-correlation-from-a-signal-analysis)

[Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

[CS231n Winter 2016: Lecture 13: Segmentation, soft attention, spatial transformers](https://www.youtube.com/watch?v=ByjaPdWXKJ4)

Semantic Segmentation
Multi-Scale testing

Iterative refinement

changed the original VGG-16, what they call conv7  is actually convolutionalized fc7 - see https://arxiv.org/pdf/1411.4038.pdf page 5.

By looking at the pre-trained weights, the pretrained VGG16 has already had the 1x1 convolution layers
'

Using tensorboard:
 tensorboard --logdir=log/

 localhost:6006

Implementation references:

[Marvin Teichmann FCN8 VGG](https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn8_vgg.py)

[VOC fcn8s](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py)
