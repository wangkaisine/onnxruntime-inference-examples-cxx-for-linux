# onnxruntime-inference-examples-cxx-for-linux
ONNX Runtime C++ sample code that can run in Linux.

I noticed that many people using ONNXRuntime wanted to see examples of code that would compile and run on Linux, so I set up this respository. The code structure of [onnxrun-time inference-examples](https://github.com/microsoft/onnxruntime-inference-examples) is kept, of course, only the parts related to C++ are kept for simplicity.

I implemented two projects using almost similar code structures and styles.

### fns_candy_style_transfer

The original fns_candy_style_transfer project is retained, because it can be compiled and run on compatible Windows platform and Linux platform, but it is a project coded by pure C language, I changed it to C++ to realize [fns_candy_style_transfer_cxx](https://github.com/wangkaisine/onnxruntime-inference-examples-cxx-for-linux/tree/main/c_cxx/fns_candy_style_transfer_cxx).

### MNIST

The original MNIST was removed because it relied heavily on components of Windows desktop programs, although it was convenient for interactive testing of true handwriting recognition. But the implementation of this code structure distracts from the core capabilities of onnxruntime, see at [MNIST_cxx](https://github.com/wangkaisine/onnxruntime-inference-examples-cxx-for-linux/tree/main/c_cxx/MNIST_cxx).


### License

This repository is under the MIT License as same as onnxruntime-inference-examples.
