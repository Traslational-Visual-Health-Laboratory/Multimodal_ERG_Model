import numpy as np
import tensorflow as tf
from keras import backend as K, Model
from keras.callbacks import Callback
from keras.regularizers import l2
from keras.layers import (Input, Dense, Dropout, Concatenate, Multiply,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalAveragePooling2D,
    Flatten, Reshape, Lambda, BatchNormalization, LayerNormalization,
    MultiHeadAttention, GRU, TimeDistributed)


############### CLASSES
class FFT_layer(tf.keras.layers.Layer):
    def __init__(self,
                 ventana_size,
                 freq_muestreo=200,
                 freq_max=40,
                 name="fft_atencion_layer",
                 value_channels=8,
                 attn_channels=8,
                 delta_init_std=1e-3,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.ventana_size = int(ventana_size)
        self.freq_muestreo = float(freq_muestreo)
        self.freq_max = float(freq_max)
        self._name = name

        self.indice_max = int(np.floor(self.freq_max * self.ventana_size / self.freq_muestreo)) + 1
        if self.indice_max < 2:
            self.indice_max = 2

        self.attn_channels = int(attn_channels)
        self.value_channels = int(value_channels)

        self.delta_init_std = float(delta_init_std)

        self.conv_query = Conv1D(self.attn_channels, kernel_size=1, padding="same", name=f"{self._name}_query")
        self.conv_key = Conv1D(self.attn_channels, kernel_size=1, padding="same", name=f"{self._name}_key")
        self.conv_value = Conv1D(self.value_channels, kernel_size=1, padding="same", name=f"{self._name}_value")

        self.conv_residual = Conv1D(self.value_channels, kernel_size=1, padding="same", name=f"{self._name}_res_proj")

        self.ln_pre = LayerNormalization(name=f"{self._name}_ln_pre")
        self.ln_post = LayerNormalization(name=f"{self._name}_ln_post")

        self.subband_ranges = [(0.3,4), (4,8), (8,12), (13,30), (30, freq_max)]

    def build(self, input_shape):
        self.delta_f = self.add_weight(
            shape=(self.indice_max,),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.delta_init_std),
            trainable=True,
            name=f"{self._name}_delta_f"
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        # inputs: (batch, ventana_size, 1) o (batch, ventana_size)
        x = tf.squeeze(inputs, -1) if inputs.shape.rank == 3 and inputs.shape[-1] == 1 else inputs
        x = tf.cast(x, tf.float32)

        fft_complex = tf.signal.rfft(x)
        fft_mag = tf.abs(fft_complex)
        fft_mag = fft_mag[:, :self.indice_max]   
        fft_real = tf.expand_dims(fft_mag, -1)

        N = tf.cast(self.ventana_size, tf.float32)
        n = tf.cast(tf.range(self.ventana_size), tf.float32)
        k = tf.cast(tf.range(self.indice_max), tf.float32)
        f_base = k * (self.freq_muestreo / self.ventana_size)

        f_learned = f_base + self.delta_f 

        cos_mat = tf.cos(2.0 * np.pi * tf.reshape(f_learned, (-1,1)) * tf.reshape(n, (1,-1)))
        sin_mat = tf.sin(2.0 * np.pi * tf.reshape(f_learned, (-1,1)) * tf.reshape(n, (1,-1)))
        cos_mat = tf.expand_dims(cos_mat, 0)
        sin_mat = tf.expand_dims(sin_mat, 0)

        x_exp = tf.expand_dims(x, 1)
        real_part = tf.reduce_sum(x_exp * cos_mat, axis=-1)
        imag_part = tf.reduce_sum(x_exp * sin_mat, axis=-1)
        fft_learned_mag = tf.sqrt(real_part**2 + imag_part**2 + 1e-8)

        fft_learned = tf.expand_dims(fft_learned_mag, -1)

        fusion = tf.concat([fft_real, fft_learned], axis=-1)

        x_norm = self.ln_pre(fusion)
        query = self.conv_query(x_norm)
        key = self.conv_key(x_norm)
        value = self.conv_value(x_norm)

        scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.cast(self.attn_channels, tf.float32) + 1e-9)
        weights = tf.nn.softmax(scores, axis=-1)

        weighted = tf.matmul(weights, value)

        res = self.conv_residual(x_norm)
        x_attn = self.ln_post(res + weighted)

        mag = tf.reduce_mean(tf.abs(x_attn), axis=-1)

        avg_pool = tf.reduce_mean(mag, axis=1, keepdims=True)
        std_pool = tf.math.reduce_std(mag, axis=1, keepdims=True)
        energy_pool = tf.reduce_sum(mag**2, axis=1, keepdims=True)

        top_k_values, top_k_indices = tf.math.top_k(mag, k=5, sorted=True)
        top_k_freqs = tf.gather(f_learned, top_k_indices)

        band_means = []
        freqs_expand = tf.reshape(f_learned, (1, -1))

        for f_low, f_high in self.subband_ranges:
            mask = tf.logical_and(freqs_expand >= f_low, freqs_expand < f_high)
            mask_f = tf.cast(mask, tf.float32)
            band_sum = tf.reduce_sum(mag * mask_f, axis=1, keepdims=True)
            band_count = tf.reduce_sum(mask_f)
            band_means.append(band_sum / tf.maximum(band_count, 1.0))

        band_means = tf.concat(band_means, axis=-1)
        output = tf.concat([
            avg_pool,
            std_pool,
            energy_pool,
            band_means,
            tf.cast(top_k_values, tf.float32),
            tf.cast(top_k_freqs, tf.float32)
        ], axis=-1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 18)

    def get_config(self):
        config = super().get_config()
        config.update({
            "ventana_size": int(self.ventana_size),
            "freq_muestreo": float(self.freq_muestreo),
            "freq_max": float(self.freq_max),
            "attn_channels": self.attn_channels,
            "value_channels": self.value_channels,
            "delta_init_std": self.delta_init_std,
            "name": self._name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class CustomEarlyStopping(Callback):
    def __init__(self, patience=10):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.best_weights = None
        self.best_val_accuracy = -np.Inf
        self.best_val_loss = np.Inf
        self.wait_accuracy = 0
        self.wait_loss = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_accuracy = logs.get('val_accuracy')
        current_val_loss = logs.get('val_loss')

        if current_val_accuracy > self.best_val_accuracy or current_val_loss < self.best_val_loss:
            if current_val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = current_val_accuracy
                self.wait_accuracy = 0
            else:
                self.wait_accuracy += 1

            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.wait_loss = 0
            else:
                self.wait_loss += 1

            self.best_weights = self.model.get_weights()
        else:
            self.wait_accuracy += 1
            self.wait_loss += 1

        if self.wait_accuracy >= self.patience and self.wait_loss >= self.patience:
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)
            print(f"\nEpoch {epoch + 1}: early stopping")

class ReduceLROnPlateauConUmbral(Callback):
    def __init__(self, monitor='val_loss', threshold=0.15, factor=0.2, patience=5, min_lr=1e-5, verbose=1):
        super().__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.wait = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        if current <= self.threshold:
            if current < self.best - 1e-4:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = max(old_lr * self.factor, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose:
                            print(f"\n🔽 ReduceLROnPlateauConUmbral: Reducción del LR a {new_lr:.6f}")
                    self.wait = 0

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
