from atten_func import *
from tools import layers as custom_layers
from models import Network
import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
layers = tf.keras.layers
models = tf.keras.models
backend = tf.keras.backend


class DeepLabV3Plus(Network):
    def __init__(self, num_classes, version='DeepLabV3Plus', base_model='Xception-DeepLab', **kwargs):
        """
        The initialization of DeepLabV3Plus.
        :param num_classes: the number of predicted classes.
        :param version: 'DeepLabV3Plus'
        :param base_model: the backbone model
        :param kwargs: other parameters
        """
        dilation = [1, 2]
        base_model = 'Xception-DeepLab' if base_model is None else base_model

        assert version == 'DeepLabV3Plus'
        assert base_model in ['VGG16',
                              'VGG19',
                              'ResNet50',
                              'ResNet101',
                              'ResNet152',
                              'DenseNet121',
                              'DenseNet169',
                              'DenseNet201',
                              'DenseNet264',
                              'MobileNetV1',
                              'MobileNetV2',
                              'Xception-DeepLab']
        super(DeepLabV3Plus, self).__init__(num_classes, version, base_model, dilation, **kwargs)
        self.dilation = dilation

    def __call__(self, inputs=None, input_size=None, **kwargs):
        assert inputs is not None or input_size is not None

        if inputs is None:
            assert isinstance(input_size, tuple)
            inputs = layers.Input(shape=input_size + (3,))
        return self._deeplab_v3_plus(inputs)

    def _deeplab_v3_plus(self, inputs):
        num_classes = self.num_classes
        _, h, w, _ = backend.int_shape(inputs)
        self.aspp_size = (h, w)
        if self.base_model in ['VGG16',
                               'VGG19',
                               'ResNet50',
                               'ResNet101',
                               'ResNet152',
                               'MobileNetV1',
                               'MobileNetV2']:
            c2, c5 = self.encoder(inputs, output_stages=['c2', 'c5'])
        else:
            c2, c5 = self.encoder(inputs, output_stages=['c1', 'c5'])

        c5 = layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(c5)
        c5 = self._conv_bn_relu(c5, 512, 9, 1)
        c5 = layers.ReLU()(c5)
        c5 = self._conv_bn_relu(c5, 256, 7, 1)
        c5 = layers.ReLU()(c5)
        c5 = self._conv_bn_relu(c5, 512, 5, 1)
        c5 = layers.ReLU()(c5)
        c5 = self._conv_bn_relu(c5, 1024, 3, 1)
        
        x = self._aspp(c5, 256)
        x = layers.Dropout(rate=0.5)(x)
               
        attention = self.cfam_module(c5,2,256,128)
        x = layers.Concatenate()([x, attention])
        
        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
        x = self._conv_bn_relu(x, 48, 1, strides=1)
        
        c2 = layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(c2)
        c2 = self._conv_bn_relu(c2, 128, 9, 1)
        c2 = layers.ReLU()(c2)
        c2 = self._conv_bn_relu(c2, 256, 7, 1)
        c2 = layers.ReLU()(c2)
        c2 = self._conv_bn_relu(c2, 128, 5, 1)
        c2 = layers.ReLU()(c2)
        c2 = self._conv_bn_relu(c2, 64, 3, 1)

        x = layers.Concatenate()([x, c2])
        x = self._conv_bn_relu(x, 256, 3, 1)
        x = layers.Dropout(rate=0.5)(x)

        x = self._conv_bn_relu(x, 256, 3, 1)
        x = layers.Dropout(rate=0.1)(x)

        x = layers.Conv2D(num_classes, 1, strides=1)(x)
        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

        outputs = layers.Activation('softmax', name='softmax_1')(x)
        return models.Model(inputs, outputs, name=self.version)

    def _conv_bn_relu(self, x, filters, kernel_size, strides=1):
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def _aspp(self, x, out_filters):
        xs = list()
        x1 = layers.Conv2D(out_filters, 1, strides=1)(x)
        x1 = se_block(x1) # channel attention
        x1 = sp_block_1(x1)
        xs.append(x1)

        for i in range(3):
            xi = layers.Conv2D(out_filters, 3, strides=1, padding='same', dilation_rate=6 * (i + 1))(x)
            xi = se_block(xi)  # channel attention
            xi = sp_block_1(xi)
            xs.append(xi)
        img_pool = custom_layers.GlobalAveragePooling2D(keep_dims=True)(x)
        img_pool = layers.Conv2D(out_filters, 1, 1, kernel_initializer='he_normal')(img_pool)
        img_pool = se_block(img_pool) # channel attention
        img_pool = sp_block_1(img_pool)
        img_pool = layers.UpSampling2D(size=self.aspp_size, interpolation='bilinear')(img_pool)
        xs.append(img_pool)   

        x = layers.Concatenate()(xs)
        x = layers.Conv2D(out_filters, 1, strides=1, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)

        return x
    
    def cfam_module(self, input,classes=2,channel=256,channel1=258):
        input_shape = input.get_shape().as_list()
        _,H,W,_ = input_shape
        N = classes
        C = channel
        C1 = channel1
        x = layers.Conv2D(C,3,padding='same',use_bias=False)(input)
        x1 = layers.Conv2D(C1,1,padding='same',use_bias=False)(x)
        x1 = tf.transpose(K.reshape(x1,(-1,H*W,C1)),(0,2,1))
        p = layers.Conv2D(N,1,padding='same',use_bias=False)(x)
        p1 = layers.Activation('softmax')(p)
        p1 = K.reshape(p1,(-1,H*W,N))
        A = K.batch_dot(x1,p1)
        A = layers.Activation('softmax')(A)
        p1 = tf.transpose(p1,(0,2,1))
        x2 = K.batch_dot(A,p1)
        x2 = K.reshape(tf.transpose(x2,(0,2,1)),(-1,H,W,C1))
        x2 = layers.Conv2D(C,(1,1),padding='same',use_bias=False)(x2)
        x2 = layers.BatchNormalization(epsilon=1e-3)(x2)
        x2 = layers.Activation('relu')(x2)
        x3 = layers.Concatenate()([x2,x])
        y = layers.Conv2D(C,(1,1),padding='same',use_bias=False)(x3)
        y = layers.BatchNormalization(epsilon=1e-3)(y)
        y = layers.Activation('relu')(y)

        return y
