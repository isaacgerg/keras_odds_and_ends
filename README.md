# keras odds and ends

Stacked what where autoencoders in tensorflow and keras.

mnist_swwae-tf.py, tensorflow compatible version of mnist_swwae.py from https://github.com/fchollet/keras/blob/master/examples/mnist_swwae.py

mnist_swwae-pure-tf.py, same as above but uses pure tensorflow, no keras.

mnist_swwae-pure-tf-paper.py, uses pure tensorflow but implements L2M error terms not computed in the above better matching derivation in paper [1]. 

======

FFT2D and IFFT2D layers. batchwise 2D FFT layers for image processing.

Schlick dynamic range compression method with auto brightness algorithm. [2]

[1] 
Stacked What-Where Auto-encoders
Junbo Zhao, Michael Mathieu, Ross Goroshin, Yann LeCun
https://arxiv.org/abs/1506.02351

[2]
@incollection{schlick1995quantization,
  title={Quantization techniques for visualization of high dynamic range pictures},
  author={Schlick, Christophe},
  booktitle={Photorealistic Rendering Techniques},
  pages={7--20},
  year={1995},
  publisher={Springer}
}


