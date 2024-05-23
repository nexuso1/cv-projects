# Author: nexuso1

import keras
import torch



def compile_model(args, model : keras.Model, training_steps, metrics = None):
    schedule = keras.optimizers.schedules.CosineDecay(args.lr, training_steps)
    optim = keras.optimizers.AdamW(learning_rate=schedule)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    if metrics is None:
        metrics = [
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.F1Score(average='macro')
        ]

    model.compile(
        optimizer=optim,
        loss=loss, 
        metrics=metrics
    )

def se_block(in_block, filters, activation='silu', ratio=4):
    """
    Squeeze-Excitation
    """
    x = keras.layers.GlobalAveragePooling2D('channels_last')(in_block)
    x = keras.layers.Dense(filters//ratio, activation=activation)(x)
    x = keras.layers.Dense(filters, activation='sigmoid')(x)
    return keras.layers.multiply([in_block, x])

def conv_norm_activ(in_block, filters, activation='relu', **kwargs):
    x = keras.layers.Conv2D(filters, activation=None, padding='same',**kwargs)(in_block)
    x = keras.layers.BatchNormalization()(x)
    if activation is not None:
        return keras.layers.Activation(activation)(x)
    else:
        return x

def depthwise_norm_activ(in_block, activation='silu', **kwargs):
    x = keras.layers.DepthwiseConv2D(padding='same', **kwargs)(in_block)
    x = keras.layers.BatchNormalization()(x)
    if activation is not None:
        return keras.layers.Activation(activation)(x)
    else:
        return x
    
def mbconv(in_block, in_filters, out_filters, expand_ratio, **kwargs):
    expanded_filters = in_filters*expand_ratio
    x = conv_norm_activ(in_block, filters=expanded_filters, kernel_size=1)
    x = depthwise_norm_activ(x, **kwargs)
    x = se_block(x, expanded_filters)
    x = conv_norm_activ(x, out_filters, kernel_size=1, activation=None)

    if in_filters == out_filters:
        # Residual connection
        x = keras.layers.Add()([in_block, x])
    
    return x

def dropout_block(in_block, filters, kernel_size=3, strides=1, first=False):
    x = conv_norm_activ(in_block,filters, kernel_size=kernel_size, strides=strides)
    x = keras.layers.SpatialDropout2D(0.2)(x)
    x = conv_norm_activ(x, filters, kernel_size=kernel_size)
    if first:
        downsampled = keras.layers.Conv2D(filters, 1, padding='same', strides=strides)(in_block)
        return keras.layers.Add()([downsampled, x])
    
    return keras.layers.Add()([in_block, x])

def dropout_group(in_block, filters, kernel_size, n, strides):
    x = dropout_block(in_block, filters, kernel_size=kernel_size, first=True, strides=strides)
    for _ in range(n-1):
        x = dropout_block(x, filters, kernel_size)

    return x

def fused_mbconv(in_block, in_filters, out_filters, expand_ratio, **kwargs):
    expanded_filters = in_filters*expand_ratio
    x = conv_norm_activ(in_block, filters=expanded_filters, **kwargs)
    x = se_block(x, expanded_filters)
    x = conv_norm_activ(x, out_filters, kernel_size=1, activation=None)

    if in_filters == out_filters:
        # Residual connection
        x = keras.layers.Add()([in_block, x])
    
    return x

def transpose_cna(in_block, in_filters, activation = 'relu', connected_block = None, **kwargs):

    x = keras.layers.Conv2DTranspose(in_filters, activation=None, padding='same', kernel_size=2, strides=2)(in_block)
    if connected_block is not None:
        x = keras.layers.Concatenate()([connected_block, x])
    x = keras.layers.BatchNormalization()(x)
    if activation is not None:
        return keras.layers.Activation(activation)(x)
    else:
        return x
    
def transpose_cna_3D(in_block, in_filters, activation = 'relu', connected_block = None, **kwargs):

    x = keras.layers.Conv3DTranspose(in_filters, activation=None, padding='same', kernel_size=2, strides=2)(in_block)
    if connected_block is not None:
        x = keras.layers.Concatenate()([connected_block, x])
    x = keras.layers.BatchNormalization()(x)
    if activation is not None:
        return keras.layers.Activation(activation)(x)
    else:
        return x
    
def conv_norm_activ_3D(in_block, filters, activation='relu', **kwargs):
    x = keras.layers.Conv3D(filters, activation=None, padding='same',**kwargs)(in_block)
    x = keras.layers.BatchNormalization()(x)
    if activation is not None:
        return keras.layers.Activation(activation)(x)
    else:
        return x

def up_3D(in_block, connected_block, in_filters, out_filters, n, kernel_size=3):
    # Upscaling
    x = transpose_cna_3D(in_block, in_filters, connected_block=connected_block)
    for i in range(n-1):
        y = conv_norm_activ_3D(x, out_filters, kernel_size=kernel_size)
        if i > 0:
            # Residual connection
            x = keras.layers.Add()([x, y])
        x = y
    return x

def down_3D(in_block, out_filters, n, **kwargs):
    x = conv_norm_activ_3D(in_block, out_filters, **kwargs)
    for i in range(n-1):
        y = conv_norm_activ_3D(x, out_filters, **kwargs)
        if i > 0:
            # Residual connection
            x = keras.layers.Add()([x, y])
        x = y

    last_conv = x
    pool = keras.layers.AveragePooling3D(2)(x)
    return pool, last_conv

def up(in_block, connected_block, in_filters, out_filters, n, kernel_size=3):
    # Upscaling
    x = transpose_cna(in_block, in_filters, connected_block=connected_block)
    for _ in range(n-1):
        x = conv_norm_activ(x, out_filters, kernel_size=kernel_size)
        
    return x

def down(in_block, out_filters, n, **kwargs):
    x = conv_norm_activ(in_block, out_filters, **kwargs)
    for _ in range(n-1):
        x = conv_norm_activ(x, out_filters, **kwargs)
    
    last_conv = x
    pool = keras.layers.AveragePooling2D(2)(x)
    return pool, last_conv

def fusedmb_group(in_block, in_filters, out_filters, expand_ratio, kernel_size = 1, n=1, strides=(1, 1)):
    x = fused_mbconv(in_block, in_filters, out_filters, expand_ratio, kernel_size=kernel_size, strides=strides)
    for _ in range(n-1):
        x = fused_mbconv(x, out_filters, out_filters, expand_ratio, kernel_size=kernel_size)
        x = keras.layers.SpatialDropout2D(0.2)(x)
    return x

def mb_group(in_block, in_filters, out_filters, expand_ratio, kernel_size = 1, n=1, strides=(1, 1)):
    x = mbconv(in_block, in_filters, out_filters, expand_ratio, kernel_size=kernel_size, strides=strides)
    for _ in range(n-1):
        x = mbconv(x, out_filters, out_filters, expand_ratio, kernel_size=kernel_size)
        x = keras.layers.SpatialDropout2D(0.2)(x)
    return x