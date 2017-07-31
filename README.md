# Evaluating the effect of class balance on convolutional neural networks to detect positive changes in multiple sclerosis lesions
~~This repository implements the method proposed in the electronic preprint available at [Arxiv](http://arxiv.org/):~~

```
Cabezas, M., Salem, M., Valverde, S., Pareto, D., Oliver, A., Rovira, À., Lladó, X. (2017). 
Evaluating the effect of class balance on convolutional neural networks to detect longitudinal changes in multiple sclerosis lesions. 
``` 

# Overview: 

Convolutional neural networks (CNN) are being increasingly used in brain MRI image analysis for segmentation of brain tumors, tissue or pathological lesions. We have developed a novel Multiple Sclerosis (MS) white matter (WM) lesion segmentation method based on a cascade of two 3D patch-wise convolutional neural networks (CNN). The first network is trained to give a list of candidate false positive, while the second one refines this segmentation by completely unbalancing the number of negative samples

The method accepts a variable number of MRI image sequences for training (T1-w, FLAIR, PD-w, T2-w, ...), which are stacked as channels into the model. However, so far, the same number of sequences have to be used for testing. In contrast to other proposed methods, the model is trained using two cascaded networks: for the first network, a balanced training dataset is generated using all positive examples (lesion voxels) and the same number of negative samples (non-lesion voxels), ranked according to their subtraction value. The first network is then used to find the most challenging examples of the entire training distribution, ie. non-lesion voxels which have being classified as lesion with a high probability. All the false positive from this first lesion are included in the training of our second network to unbalance the classifier and bias it.


# Install:

The method works on top of [Lasagne](http://lasagne.readthedocs.io/en/latest/index.html) and [Theano](http://deeplearning.net/software/theano/). If the method is run using GPU, please be sure that the Theano ```cuda*``` backend has been installed [correctly](https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29). Make also sure that the [cuDNN files](https://developer.nvidia.com/cudnn) are also available in your ```cuda*``` installation. In the case of CPU, be sure that the fast linear algebra [libraries](http://lasagne.readthedocs.io/en/latest/user/installation.html#numpy-scipy-blas) are also installed. 

Once these requirements are met, the rest of python libraries may be easily installed using ```pip```: 

```python
pip install -r requirements.txt 
```


# How to use it: 
In order to run these tool, all the images should be on the same space. If available, deformation fields can also be used to improve the segmentation as stated in the article. You can use our [tool](https://github.com/NIC-VICOROB/braintools) to perform all of that preprocessing.

To run the train and testing, two folder must be defined and passed as parameters when running. Each folder must have each patient separated in their own subfolder. If needed, subfolder inside each patient's folder can be defined to point to the images folder, a subtraction folder (used to initialise the method but not really necessary) and the deformation folder. For the leave-one-out experiments, there is a custom [script](https://github.com/marianocabezas/cnn-nolearn/blob/master/train_test_longitudinal.py) used in our experiments. This script is part of the author's CNN sandbox repository using lasagne and nolearn.

# Citing this work:

No citation yet.

