
from models import *
import tensorflow as tf

layers = tf.keras.layers


def builder(num_classes, input_size=(256, 256), model='DeepLabV3Plus', base_model=None):
    models = {'FCN-8s': FCN,
              'FCN-16s': FCN,
              'FCN-32s': FCN,
              'SegNet': SegNet,
              'Bayesian-SegNet': SegNet,
              'PAN': PAN,
              'PSPNet': PSPNet,
              'RefineNet': RefineNet,
              'DenseASPP': DenseASPP,
              'DeepLabV3': DeepLabV3,
              'DeepLabV3Plus': DeepLabV3Plus,
              'BiSegNet': BiSegNet}

    assert model in models

    net = models[model](num_classes, model, base_model)

    inputs = layers.Input(shape=input_size+(7,))

    return net(inputs), net.get_base_model()
