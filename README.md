# PyTorch-Examples

## 01. LinearRegression:
An simple example using PyTorch to fit a linear equation.

## 02. Image Convolution:
Performing the convolution operation (blur) on an image.

## 03. CNN:
Creating a CNN classifier.

## 04. Res Net:
Using pretrained resnet18 to classify CIFAR10/MNIST images.
Using tensorboard to visualize the running loss.

## 05. Transfer Learning:
An complete example of transfer learning.

These two major transfer learning scenarios look as follows:
<ul>
  <li>Finetuning the convnet</li> 
  <li>ConvNet as fixed feature extractor</ul>
</ul>

## 06. Distributed Training:
An example to perform the communication through the process during the training process.

<ul>
  <li>point2point.py: perform p2p communication</li>
  <li>group.py: communicate with the nodes in the specific group</li>
  <li>distributed.py: partition the training data to different GPUs and reduce the gradient from each nodes</li>
</ul>