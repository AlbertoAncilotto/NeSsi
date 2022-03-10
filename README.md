# NeSsi
Keras/Pytorch neural network size, operations and parameters counter.
Relies on tensorflow profiler for tf/keras models, torchinfo (https://github.com/TylerYep/torchinfo) for torch models, and tflite tools (https://github.com/eliberis/tflite-tools) for tensorflow lite model profiling.

# Example Usage:
For tf/Keras models, the input size gets inferred from the first layer of the network, so avoid [None] undefined dimensions. Batch size gets automatically set to 1.

[Keras]: 

`import nessi`

`nessi.get_model_size(model, 'keras')`

For torch models, you need to specify the input size as a touple when calling `get_model_size`.

[Torch]: 

`import nessi`

`nessi.get_model_size(net, 'torch' ,input_size=(1,3,320,320))`

For tflite models, either pass the model path or the bytearray to `get_model_size`. 

[TFLite]: 

`import nessi`

`nessi.get_model_size(tflite_model OR path/to/model.tflite, 'tflite')`

