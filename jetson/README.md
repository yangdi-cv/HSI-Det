# HSI-Det on Jetson

`weights/hsidet.onnx` is the HSI-Det-B model exported with ONNX opset 17.
It has a fixed input shape of `1 x 25 x 640 x 640` and a raw output shape of
`1 x 27 x 8400`. The ONNX model does not include non-maximum suppression.

Copy `weights/hsidet.onnx` to the Jetson. From the directory containing the
file, build an FP16 TensorRT engine:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=hsidet.onnx \
  --saveEngine=hsidet.engine \
  --fp16
```

Build the engine on the target Jetson. Apply the same 25-band input
preprocessing and detection postprocessing used for the ONNX model. TensorRT
conversion has not yet been tested on a Jetson in this project.
