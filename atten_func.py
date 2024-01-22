import tensorflow as tf
layers = tf.keras.layers
backend = tf.keras.backend


# Channel attention, 来自于SEnet paper
def se_block(x):
    bs, h, w, c = x.shape.as_list()
    gap_x = layers.GlobalAveragePooling2D()(x)  # [N, C]
    fc1_x = layers.Dense(units=c//16, activation='relu',use_bias=True)(gap_x)  # [N, C/16] ,
    #当输入的特征通道数量比较大时，如512， 这时可以讲16改为32. 可以将16当做一个int参数由函数传进来
    fc2_x = layers.Dense(units=c, activation='sigmoid')(fc1_x) # [N, C]
    excitation_x = layers.Reshape((1,1,c))(fc2_x)  # [N, 1, 1, C]
    scale_x = layers.Multiply()([x, excitation_x]) # [N, H, W, C]
    out = layers.Add()([x, scale_x])  # [N, H, W, C]

    return out


# Spatial attention:  第1种，来自于 CBMA(Convolutional Block Attention Module) paper
def sp_block_1(x):

    avg_pool = layers.Lambda(lambda x: backend.mean(x, axis=3, keepdims=True))(x) # [N, H, W, 1]
    max_pool = layers.Lambda(lambda x: backend.max(x, axis=3, keepdims=True))(x) # [N, H, W, 1]

    concat = layers.Concatenate(axis=3)([avg_pool, max_pool]) # [N, H, W, 2]

    cbam_feature = layers.Conv2D(filters=1,
                          kernel_size=7,
                          activation='sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=True)(concat)
                                        # [N, H, W, 1]

    out = layers.Multiply()([x, cbam_feature])

    return out


# Spatial attention:  第2种，来自于 paper
# Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
def sp_block_2(x):
    squeeze_x = layers.Conv2D(filters=1,
                        kernel_size=1,
                        activation='sigmoid',
                        strides=1,
                        padding='same',
                        kernel_initializer='he_normal',
                        use_bias=True)(x)
                                        # [N, H, W, 1]

    excitation_x = layers.Multiply()([x, squeeze_x])

    return excitation_x
