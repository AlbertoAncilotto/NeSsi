import importlib
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

MAX_MACC=30e6       #30M MACC
MAX_PARAMS=128e4    #128K params


def get_keras_size(model):
    #we save the model to disk and reset the graph to avoid stacking the macc count from the profiler
    params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    config = model.get_config() # Returns pretty much every information about your model
    in_sz=np.array(config["layers"][0]["config"]["batch_input_shape"])
    in_sz[in_sz==None]=1
    print(in_sz)
    model.save('tmp.h5')
    tf.compat.v1.reset_default_graph()
    macc = get_maccs('tmp.h5', in_sz)
    return macc, params

def get_maccs(model_h5_path, in_size):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)(np.ones(in_size))
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
            return flops.total_float_ops//2

def get_torch_size(model, input_size):
    import torchinfo
    model_profile= torchinfo.summary(model, input_size=input_size)
    return model_profile.total_mult_adds, model_profile.total_params

def get_tflite_size(model):
    import tflite_tools as tflt

    if isinstance(model, bytes):
        model = tflt.TFLiteModel.load_from_bytes(model)
    else:
        model = tflt.TFLiteModel.load_from_file(model)
    macc, params = model.print_model_analysis(macs=True, size=True)
    return macc, params

def get_model_size(model, model_type='keras', input_size=None):
    if model_type=='keras':
        macc, params=get_keras_size(model)
    elif model_type=='torch':
        macc, params=get_torch_size(model, input_size)
    elif model_type=='tflite':
        macc, params=get_tflite_size(model)
    else:
        print('type', model_type, 'not supported, possibilities: ["keras", "torch", "tflite"]')
    validate(macc, params)
    
def validate(macc, params):
    print('Model statistics:')
    print('MACC:\t \t %.3f' %  (macc/1e6), 'M')
    print('Params:\t \t %.3f' %  (params/1e3), 'K\n')
    if macc>MAX_MACC:
        print('[Warning] Multiply accumulate count', macc, 'is more than the allowed maximum of', int(MAX_MACC))
    if params>MAX_PARAMS:
        print('[Warning] parameter count', params, 'is more than the allowed maximum of', int(MAX_PARAMS))