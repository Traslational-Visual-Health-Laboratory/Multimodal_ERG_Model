import tensorflow as tf
from keras import backend as Model
from keras.regularizers import l2
from classes import *
from keras.layers import (Input, Dense, Dropout, Concatenate, Multiply,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalAveragePooling2D,
    Flatten, Reshape, Lambda, BatchNormalization, LayerNormalization,
    MultiHeadAttention, GRU, TimeDistributed)


############## BRANCHES

def transformer(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dropout(dropout)(ff)
    ff = Dense(inputs.shape[-1])(ff)
    return LayerNormalization(epsilon=1e-6)(ff + x)

def conv_block(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = GlobalAveragePooling1D()(x)
    model = Model(inputs, x)
    return model

def conv_block2(inputs, l2_lambda):
    conv5 = Conv1D(32, 5, padding='same', activation='elu',
                   kernel_regularizer=l2(l2_lambda))(inputs)
    
    conv25 = Conv1D(16, 25, padding='same', activation='elu',
                    kernel_regularizer=l2(l2_lambda))(inputs)
    
    conv100 = Conv1D(8, 100, padding='same', activation='elu',
                     kernel_regularizer=l2(l2_lambda))(inputs)

    # Concatenar resultados
    x = Concatenate()([conv5, conv25, conv100])
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    return x

def conv_block_final(input_shape, l2_lambda=1e-4, dropout_rate=0.2):
    inputs = Input(shape=input_shape)
    x = conv_block2(inputs, l2_lambda)
    x = GRU(32, return_sequences=True,
            kernel_regularizer=l2(l2_lambda),
            recurrent_regularizer=l2(l2_lambda))(x)
    x = Dropout(dropout_rate)(x)
    attention = GlobalAveragePooling1D()(x)
    attention = Dense(32, activation="elu", kernel_regularizer=l2(l2_lambda))(attention)
    attention = Dense(x.shape[-1], activation="sigmoid", kernel_regularizer=l2(l2_lambda))(attention)
    attention = Lambda(lambda a: tf.expand_dims(a, axis=1))(attention)
    x = Multiply()([x, attention])
    x = Dropout(dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)

    model = Model(inputs, x)
    return model

def FFT_block(input_shape, freq_muestreo=200, freq_max=40, name="fft_layer"):
    return FFT_layer(
        ventana_size=input_shape[0],
        freq_muestreo=freq_muestreo,
        freq_max=freq_max,
        name=name
    )

############### MODELS

def multimodal_erg_model(input_shape, img_size, num_clases, window_size, num_variables_adicionales,
                             d_model=128, num_heads=4, dropout=0.2):
    inputs_signal = Input(shape=(None, window_size, input_shape[-1]))
    inputs_img = Input(shape=(img_size[0], img_size[1], 3))
    inputs_clinicos = Input(shape=(num_variables_adicionales,))

    x_cnn = TimeDistributed(conv_block_final((window_size, input_shape[-1])))(inputs_signal)
    x_cnn = TimeDistributed(BatchNormalization())(x_cnn)

    x_fft = TimeDistributed(FFT_block((window_size, input_shape[-1])))(inputs_signal)
    x_fft = LayerNormalization(epsilon=1e-6)(x_fft)

    x_cnn = transformer(x_cnn, head_size=64, num_heads=4, ff_dim=128, dropout=dropout)
    x_fft = transformer(x_fft, head_size=64, num_heads=4, ff_dim=128, dropout=dropout)

    fft_to_cnn = MultiHeadAttention(num_heads=4, key_dim=64, dropout=dropout)(query=x_fft, key=x_cnn, value=x_cnn)
    cnn_to_fft = MultiHeadAttention(num_heads=4, key_dim=64, dropout=dropout)(query=x_cnn, key=x_fft, value=x_fft)

    x_fft = LayerNormalization()(x_fft + fft_to_cnn)
    x_cnn = LayerNormalization()(x_cnn + cnn_to_fft)
    x_signal = Concatenate()([x_fft, x_cnn])

    base_model = tf.keras.applications.VGG16(include_top=False, weights="imagenet",
                                             input_shape=(img_size[0], img_size[1], 3))
    base_model.trainable = False
    x_img = base_model(inputs_img)
    x_img = GlobalAveragePooling2D()(x_img)
    x_img = Flatten()(x_img)
    x_img = BatchNormalization()(x_img)

    flatten_shape = x_img.shape[1]
    num_patches = flatten_shape // d_model
    if flatten_shape % d_model != 0:
        raise ValueError("El tamaño de entrada no es divisible por d_model")

    x_img = Reshape((num_patches, d_model))(x_img)

    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x_img, x_img)
    x_img = LayerNormalization()(x_img + attention_output)
    ffn_output = Dense(512, activation='relu')(x_img)
    ffn_output = Dense(d_model)(ffn_output)
    x_img = LayerNormalization()(x_img + ffn_output)
    x_img = tf.reduce_mean(x_img, axis=1)
    x_img = tf.expand_dims(x_img, axis=1)
    x_img = tf.repeat(x_img, repeats=tf.shape(x_signal)[1], axis=1)

    signal_to_img = MultiHeadAttention(num_heads=4, key_dim=64, dropout=dropout)(query=x_signal, key=x_img, value=x_img)
    img_to_signal = MultiHeadAttention(num_heads=4, key_dim=64, dropout=dropout)(query=x_img, key=x_signal, value=x_signal)

    x_signal = LayerNormalization()(x_signal + signal_to_img)
    x_img = LayerNormalization()(x_img + img_to_signal)

    x_multimodal = Concatenate()([x_signal, x_img])

    x_clinicos = Dense(128, activation="relu")(inputs_clinicos)
    x_clinicos = Dropout(dropout)(x_clinicos)
    x_clinicos = Dense(64, activation="relu")(x_clinicos)
    x_clinicos = tf.expand_dims(x_clinicos, axis=1)

    clinica_to_modal = MultiHeadAttention(num_heads=4, key_dim=64, dropout=dropout)(
        query=x_clinicos, value=x_multimodal, key=x_multimodal
    )
    modal_to_clinica = MultiHeadAttention(num_heads=4, key_dim=64, dropout=dropout)(
        query=x_multimodal, value=x_clinicos, key=x_clinicos
    )

    clinica_to_modal = tf.squeeze(clinica_to_modal, axis=1)
    modal_to_clinica = tf.reduce_mean(modal_to_clinica, axis=1)
    x_fusion = Concatenate()([clinica_to_modal, modal_to_clinica])

    x = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x_fusion)
    x = Dropout(dropout)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout)(x)

    outputs = Dense(num_clases, activation="sigmoid" if num_clases == 1 else "softmax")(x)
    model = Model(inputs=[inputs_signal, inputs_clinicos, inputs_img], outputs=outputs, name="modelo_multimodal2")
    return model

def img_model(input_shape, img_size, num_clases, window_size, num_variables_adicionales, d_model=128, num_heads=4):
    inputs = Input(shape=(None, window_size, input_shape[-1]))
    inputs_adicionales = Input(shape=(num_variables_adicionales,))
    inputs_img = Input(shape=(img_size[0], img_size[1], 3))
    x_cnn = tf.zeros((tf.shape(inputs)[0], tf.shape(inputs)[1], d_model))
    x_fft = tf.zeros((tf.shape(inputs)[0], tf.shape(inputs)[1], d_model))
    x_signal = Concatenate()([x_fft, x_cnn])
    x_signal = Dense(d_model, activation="relu")(x_signal)
    base_model = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(img_size[0], img_size[1], 3))
    base_model.trainable = False
    x_img = base_model(inputs_img)
    x_img = GlobalAveragePooling2D()(x_img)
    x_img = Flatten()(x_img)
    x_img = BatchNormalization()(x_img)
    flatten_shape = x_img.shape[1]
    num_patches = flatten_shape // d_model

    if flatten_shape % d_model != 0:
        raise ValueError("Size is not compatible with d_model")

    x_img = Reshape((num_patches, d_model))(x_img)
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x_img, x_img)
    x_img = LayerNormalization()(x_img + attention_output)
    ffn_output = Dense(512, activation='relu')(x_img)
    ffn_output = Dense(d_model)(ffn_output)
    x_img = LayerNormalization()(x_img + ffn_output)
    x_img = tf.reduce_mean(x_img, axis=1)
    x_img = tf.expand_dims(x_img, axis=1)
    x_img = tf.repeat(x_img, repeats=tf.shape(x_signal)[1], axis=1)
    x_combined = Concatenate()([x_signal, x_img])
    x_adicionales = Dense(128, activation="relu")(inputs_adicionales)
    x_adicionales = tf.expand_dims(x_adicionales, axis=1)
    attention_output, _ = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)(
        query=x_adicionales, value=x_combined, key=x_combined, return_attention_scores=True
    )
    x_fusion = tf.squeeze(attention_output, axis=1)
    x_fusion = Concatenate()([x_fusion, x_adicionales[:, 0, :]])

    x = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x_fusion)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)

    outputs = Dense(num_clases, activation="sigmoid" if num_clases == 1 else "softmax")(x)

    model = Model(inputs=[inputs, inputs_adicionales, inputs_img], outputs=outputs)
    return model

def create_vit(
    input_shape=(224,224,3),
    patch_size=16,
    d_model=128,
    num_heads=4,
    num_layers=4,
    num_classes=2,
    dropout=0.1
):
    inputs = Input(shape=input_shape)

    x = PatchEmbedding(patch_size, d_model)(inputs)
    num_patches = (input_shape[0] // patch_size) ** 2
    x = CLSToken(d_model)(x)
    x = PositionalEmbedding(num_patches + 1, d_model)(x)
    for _ in range(num_layers):
        x = TransformerBlock(d_model, num_heads, dropout)(x)
    x = x[:, 0]

    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)

def vit_multimodal(input_shape, img_size, num_classes, window_size, num_variables_adicionales,
                         patch_size=16, d_model=128, num_heads=4, mlp_dim=512, dropout=0.2):
    inputs = Input(shape=(None, window_size, input_shape[-1]), name="input_signal")
    inputs_adicionales = Input(shape=(num_variables_adicionales,), name="input_clinical")
    inputs_img = Input(shape=(img_size[0], img_size[1], 3), name="input_image")
    x_cnn = tf.zeros((tf.shape(inputs)[0], tf.shape(inputs)[1], d_model))
    x_fft = tf.zeros((tf.shape(inputs)[0], tf.shape(inputs)[1], d_model))
    x_signal = Concatenate()([x_fft, x_cnn])
    x_signal = Dense(d_model, activation="relu")(x_signal)

    patches = tf.image.extract_patches(
        images=inputs_img,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    batch_size = tf.shape(patches)[0]
    num_patches_H = img_size[0] // patch_size
    num_patches_W = img_size[1] // patch_size
    num_patches = num_patches_H * num_patches_W
    patch_dim = patch_size * patch_size * 3

    x_img = tf.reshape(patches, (batch_size, num_patches, patch_dim))
    x_img = Dense(d_model)(x_img)

    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=d_model)(positions)
    x_img = x_img + pos_embed

    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x_img, x_img)
    x_img = LayerNormalization()(x_img + attn_output)
    ffn_output = Dense(mlp_dim, activation='relu')(x_img)
    ffn_output = Dense(d_model)(ffn_output)
    x_img = LayerNormalization()(x_img + ffn_output)
    x_img = tf.reduce_mean(x_img, axis=1)
    x_img = tf.expand_dims(x_img, axis=1)
    x_img = tf.repeat(x_img, repeats=tf.shape(x_signal)[1], axis=1)
    x_combined = Concatenate()([x_signal, x_img])

    x_ad = Dense(128, activation="relu")(inputs_adicionales)
    x_ad = tf.expand_dims(x_ad, axis=1)
    attention_output, _ = MultiHeadAttention(num_heads=4, key_dim=64, dropout=dropout)(
        query=x_ad, value=x_combined, key=x_combined, return_attention_scores=True
    )
    x_fusion = tf.squeeze(attention_output, axis=1)
    x_fusion = Concatenate()([x_fusion, x_ad[:, 0, :]])

    x = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x_fusion)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)

    outputs = Dense(num_classes, activation="sigmoid" if num_classes == 1 else "softmax")(x)

    model = Model(inputs=[inputs, inputs_adicionales, inputs_img], outputs=outputs, name="ViT_multimodal_silenced_signal")
    return model

def multimodal_img_model(input_shape, img_size, num_clases, window_size,
                                    num_variables_adicionales, d_model=128, num_heads=4, dropout=0.2):
    inputs_signal = Input(shape=(None, window_size, input_shape[-1]))
    inputs_img = Input(shape=(img_size[0], img_size[1], 3))
    inputs_clinicos = Input(shape=(num_variables_adicionales,))

    x_signal = tf.zeros((tf.shape(inputs_signal)[0], tf.shape(inputs_signal)[1], d_model))

    base_model = tf.keras.applications.VGG16(include_top=False, weights="imagenet",
                                             input_shape=(img_size[0], img_size[1], 3))
    base_model.trainable = False

    x_img = base_model(inputs_img) 
    x_img = GlobalAveragePooling2D()(x_img) 
    x_img = Dense(d_model, activation="relu")(x_img)
    x_img = tf.expand_dims(x_img, axis=1)
    x_img = tf.repeat(x_img, repeats=tf.shape(x_signal)[1], axis=1)

    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x_img, x_img)
    x_img = LayerNormalization()(x_img + attention_output)
    ffn_output = Dense(512, activation='relu')(x_img)
    ffn_output = Dense(d_model)(ffn_output)
    x_img = LayerNormalization()(x_img + ffn_output)

    signal_to_img = MultiHeadAttention(num_heads=4, key_dim=d_model, dropout=dropout)(
        query=x_signal, key=x_img, value=x_img
    )
    img_to_signal = MultiHeadAttention(num_heads=4, key_dim=d_model, dropout=dropout)(
        query=x_img, key=x_signal, value=x_signal
    )

    x_signal = LayerNormalization()(x_signal + signal_to_img)
    x_img = LayerNormalization()(x_img + img_to_signal)

    x_multimodal = Concatenate()([x_signal, x_img])

    x_clinicos = Dense(128, activation="relu")(inputs_clinicos)
    x_clinicos = Dropout(dropout)(x_clinicos)
    x_clinicos = Dense(64, activation="relu")(x_clinicos)
    x_clinicos = tf.expand_dims(x_clinicos, axis=1)

    clinica_to_modal = MultiHeadAttention(num_heads=4, key_dim=64, dropout=dropout)(
        query=x_clinicos, key=x_multimodal, value=x_multimodal
    )
    modal_to_clinica = MultiHeadAttention(num_heads=4, key_dim=64, dropout=dropout)(
        query=x_multimodal, key=x_clinicos, value=x_clinicos
    )

    clinica_to_modal = tf.squeeze(clinica_to_modal, axis=1)
    modal_to_clinica = tf.reduce_mean(modal_to_clinica, axis=1)
    x_fusion = Concatenate()([clinica_to_modal, modal_to_clinica])

    x = Dense(128, activation="relu")(x_fusion)
    x = Dropout(dropout)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout)(x)

    outputs = Dense(num_clases, activation="sigmoid" if num_clases == 1 else "softmax")(x)

    model = Model(inputs=[inputs_signal, inputs_clinicos, inputs_img], outputs=outputs)
    return model

def cx_model(input_shape, img_size, num_clases, window_size,
                               num_variables_adicionales, d_model=128, num_heads=4, dropout=0.2):
   
    inputs_signal = Input(shape=(None, window_size, input_shape[-1]))
    inputs_img = Input(shape=(img_size[0], img_size[1], 3))
    inputs_clinicos = Input(shape=(num_variables_adicionales,))
    x_signal = tf.zeros((tf.shape(inputs_signal)[0], tf.shape(inputs_signal)[1], d_model))
    x_img = tf.zeros((tf.shape(inputs_signal)[0], tf.shape(inputs_signal)[1], d_model))

    signal_to_img = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)(
        query=x_signal, key=x_img, value=x_img
    )
    img_to_signal = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)(
        query=x_img, key=x_signal, value=x_signal
    )

    x_signal = LayerNormalization()(x_signal + signal_to_img)
    x_img = LayerNormalization()(x_img + img_to_signal)

    x_multimodal = Concatenate()([x_signal, x_img])
    x_clinicos = Dense(128, activation="relu")(inputs_clinicos)
    x_clinicos = Dropout(dropout)(x_clinicos)
    x_clinicos = Dense(64, activation="relu")(x_clinicos)
    x_clinicos = tf.expand_dims(x_clinicos, axis=1)

    clinica_to_modal = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=dropout)(
        query=x_clinicos, key=x_multimodal, value=x_multimodal
    )
    modal_to_clinica = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=dropout)(
        query=x_multimodal, key=x_clinicos, value=x_clinicos
    )

    clinica_to_modal = tf.squeeze(clinica_to_modal, axis=1)
    modal_to_clinica = tf.reduce_mean(modal_to_clinica, axis=1)
    x_fusion = Concatenate()([clinica_to_modal, modal_to_clinica])

    x = Dense(128, activation="relu")(x_fusion)
    x = Dropout(dropout)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout)(x)

    outputs = Dense(num_clases, activation="sigmoid" if num_clases == 1 else "softmax")(x)

    model = Model(inputs=[inputs_signal, inputs_clinicos, inputs_img], outputs=outputs)
    return model
