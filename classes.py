import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import (Dense, Dropout, Conv1D, LayerNormalization, MultiHeadAttention, Conv2D)


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

class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.proj = Conv2D(
            filters=d_model,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid"
        )

    def call(self, images):
        x = self.proj(images)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, x.shape[-1]])
        return x

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_patches, d_model):
        super().__init__()
        self.pos_emb = self.add_weight(
            "pos_emb",
            shape=(1, num_patches, d_model),
            initializer="random_normal"
        )

    def call(self, x):
        return x + self.pos_emb

class CLSToken(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.cls = self.add_weight(
            "cls",
            shape=(1, 1, d_model),
            initializer="random_normal"
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.broadcast_to(self.cls, [batch_size, 1, x.shape[-1]])
        return tf.concat([cls_tokens, x], axis=1)
    
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.norm1 = LayerNormalization()
        self.dropout1 = Dropout(dropout)

        self.ffn = tf.keras.Sequential([
            Dense(512, activation='relu'),
            Dense(d_model)
        ])

        self.norm2 = LayerNormalization()
        self.dropout2 = Dropout(dropout)

    def call(self, x, training=False):
        attn_output = self.attn(x, x)
        x = self.norm1(x + attn_output)
        x = self.dropout1(x, training=training)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        x = self.dropout2(x, training=training)
        return x

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

