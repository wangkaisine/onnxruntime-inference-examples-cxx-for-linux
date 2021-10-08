# MNIST C++
A handwriting recognition application that runs directly from the command line, without the Windows GUI.

# Build Instructions
See [../README.md](../README.md)

# Prepare data
You can download the [MINST model](https://github.com/onnx/models/tree/master/vision/classification/mnist) here, or use the [MNIST model](https://github.com/wangkaisine/onnxruntime-inference-examples-cxx-for-linux/tree/main/resources/MNIST) in our resources.

Then, prepare an image:
1. Grayscale PNG format
2. Dimension of 28x28

# Run
Command to run the application:
```
./MNIST_cxx <model_path> <input_image_path> [cpu|cuda|dml]
```

To use the CUDA or DirectML execution providers, specify `cuda` or `dml` on the command line. `cpu` is the default.
