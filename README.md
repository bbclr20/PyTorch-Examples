# PyTorch-Examples

## 01. LinearRegression

An simple example using PyTorch to fit a linear equation.

## 02. Image Convolution

Performing the convolution operation (blur) on an image.

## 03. CNN

Creating a CNN classifier to predict MNIST dataset.

## 04. Res Net

Using pretrained **resnet18** to classify CIFAR10/MNIST images.
Using tensorboard to visualize the running loss.

## 05. Transfer Learning

An complete example of transfer learning.

These two major transfer learning scenarios look as follows:

* Finetuning the convnet
* ConvNet as fixed feature extractor

## 06. Distributed Training

An example to perform the communication through the process during the training process.

* point2point.py: perform p2p communication
* group.py: communicate with the nodes in the specific group
* distributed.py: partition the training data to different GPUs and reduce the gradient from each nodes

## 07. Spatial Transformation

**Spatial transformer networks (STN for short)** allow a neural network to learn how to perform spatial transformations
 on the input image in order to enhance the geometric invariance of the model.

## 08. ONNX

Using PyTorch to exporting the ONNX model. The model is loaded ant tested with Caffe2 backend.

## 10. Neural Style Transfer

**Neural style transfer** is an optimization technique used to take three images, a content image, a style reference image and the input image you want to style and blend them together[2].

## Reference

[1] <https://pytorch.org/tutorials/>

[2] <https://arxiv.org/abs/1508.06576>