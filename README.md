# NeSsi
Keras/Pytorch neural network size, operations and parameters counter.

# Example Usage:
For tf/Keras models, the input size gets inferred from the first layer of the network, so avoid [None] undefined dimensions

[Keras]: 

`import nessi`

`nessi.get_model_size(model, 'keras')`


[Torch]: 

`import nessi`

`nessi.get_model_size(net, 'torch' ,input_size=(1,3,320,320))`
