import os
import re
import gc
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import tensorflow as tf
import shap
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

sampling_freq = 2000
window_size   = round(sampling_freq * 15)
step_size     = round(sampling_freq * 1)

additional_columns  = ['AGE', 'SEX']
categorical_columns = ['SEX']
numeric_columns     = ['AGE']

data_path        = r''
train_data_path  = r''
model_path       = r""
output_path      = r''
master_file      = r''
backbone         = "VGG16"

# Scalogram frequencies (logarithmic Y axis, high frequency at the top)
f_min = 0.3
f_max = 40.0
bands = {
    "0.3-2 Hz":  (0.3,  2.0),
    "2-4 Hz":    (2.0,  4.0),
    "4-8 Hz":    (4.0,  8.0),
    "8-13 Hz":   (8.0, 13.0),
    "13-30 Hz": (13.0, 30.0),
    "30-40 Hz": (30.0, 40.0),
}


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM LAYER  (needed to load the model)
# ══════════════════════════════════════════════════════════════════════════════

@tf.keras.utils.register_keras_serializable(package='CustomLayers')
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

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (same logic as val_PAPER_SHAP.py)
# ══════════════════════════════════════════════════════════════════════════════

model_preprocessors = {
    "DenseNet201":        tf.keras.applications.densenet.preprocess_input,
    "MobileNetV2":        tf.keras.applications.mobilenet_v2.preprocess_input,
    "ResNet50":           tf.keras.applications.resnet50.preprocess_input,
    "InceptionV3":        tf.keras.applications.inception_v3.preprocess_input,
    "Xception":           tf.keras.applications.xception.preprocess_input,
    "VGG16":              tf.keras.applications.vgg16.preprocess_input,
    "VGG19":              tf.keras.applications.vgg19.preprocess_input,
    "ResNet101":          tf.keras.applications.resnet.preprocess_input,
    "InceptionResNetV21": tf.keras.applications.inception_resnet_v2.preprocess_input,
    "MobileNet":          tf.keras.applications.mobilenet.preprocess_input,
    "EfficientNetB0":     tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetB7":     tf.keras.applications.efficientnet.preprocess_input,
    "ResNet50V2":         tf.keras.applications.resnet_v2.preprocess_input,
    "NASNetLarge":        tf.keras.applications.nasnet.preprocess_input,
}


def load_trained_model(model_path):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"FFT_layer": FFT_layer}
    )

def load_additional_data(master_path):
    return pd.read_excel(master_path)

def get_additional_data(key, master_df, columns):
    row = master_df[master_df['CLAVE'] == key]
    if not row.empty:
        return row[columns].values.flatten()
    return np.zeros(len(columns))

def normalize_additional_data(df, categorical_columns, numeric_columns):
    for col in categorical_columns:
        if len(df[col].unique()) == 2:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        else:
            ohe = OneHotEncoder(sparse=False)
            encoded_cols = ohe.fit_transform(df[[col]])
            encoded_df   = pd.DataFrame(
                encoded_cols,
                columns=[f"{col}_{cat}" for cat in ohe.categories_[0]]
            )
            df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

def split_into_windows(signal, window_size, step_size):
    windows = []
    for start in range(0, signal.shape[0] - window_size + 1, step_size):
        windows.append(signal[start:start + window_size])
    return np.stack(windows, axis=0)

def preprocess_data(
    path, window_size, step_size,
    scales=None, master_path=None,
    additional_columns=None,
    categorical_columns=None,
    numeric_columns=None,
    img_size=(224, 224),
    img_preprocessor=None,
):
    master_df = load_additional_data(master_path) if master_path else None
    if master_df is not None and categorical_columns and numeric_columns:
        master_df = normalize_additional_data(
            master_df, categorical_columns, numeric_columns
        )

    classes     = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    num_classes = len(classes)

    time_series       = []
    scalograms        = []
    labels            = []
    additional_data   = []
    patient_keys      = []

    original_img_paths = []

    for idx, class_name in enumerate(classes):
        folder    = os.path.join(path, class_name)
        csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]

        for csv_file in csv_files:
            # ── Patient key ──
            file_name = os.path.basename(csv_file)
            key_str   = file_name.split('_')[0]
            key_str   = re.search(r'\d+', key_str).group(0)
            key_int   = int(key_str)

            # ── Additional data ──
            additional_data_row = (
                get_additional_data(key_int, master_df, additional_columns)
                if master_df is not None
                else np.zeros(len(additional_columns))
            )
            additional_data.append(additional_data_row)

            # ── Time signal ──
            csv_path = os.path.join(folder, csv_file)
            df       = pd.read_csv(csv_path)
            signal   = df.values
            windows  = split_into_windows(signal, window_size, step_size)
            time_series.append(windows)

            # ── Scalogram ──
            img_file = csv_file.replace('.csv', '.png')
            img_path = os.path.join(folder, img_file)
            original_img_paths.append(img_path if os.path.exists(img_path) else None)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                if img_preprocessor is not None:
                    img = img_preprocessor(img.astype(np.float32))
                else:
                    img = img / 255.0
                scalograms.append(img)
            else:
                # Placeholder if the image does not exist
                scalograms.append(np.zeros((*img_size, 3), dtype=np.float32))

            labels.append(idx)
            patient_keys.append(key_str)

    x_series        = np.array(time_series)
    x_img           = np.array(scalograms)
    y               = np.array(labels)
    additional_data = np.array(additional_data)

    return x_series, x_img, y, additional_data, num_classes, classes, patient_keys, original_img_paths

def build_image_submodel(model, x_fixed, additional_data_fixed):
    """
    Builds a Keras image→prediction sub-model with x and additional_data
    embedded as Lambda constants. Compatible with GradientExplainer
    in eager mode (without disable_eager_execution).

    Parameters
    ----------
    model                  : multimodal model with inputs [x_series, additional_data, img]
    x_fixed                : 2-D array (window_size, n_channels) — signal of the current sample
    additional_data_fixed  : 1-D array (D,) — additional data of the current sample
    """
    img_shape = model.input[2].shape[1:]           # (H, W, C)
    img_input = tf.keras.Input(shape=img_shape, name="img_only")

    x_value               = x_fixed[np.newaxis].astype(np.float32)               # (1, T, C)
    additional_data_value = additional_data_fixed[np.newaxis].astype(np.float32) # (1, D)

    x_layer = tf.keras.layers.Lambda(
        lambda img: tf.repeat(
            tf.constant(x_value, dtype=tf.float32),
            tf.shape(img)[0], axis=0
        ), name="x_const"
    )(img_input)

    additional_data_layer = tf.keras.layers.Lambda(
        lambda img: tf.repeat(
            tf.constant(additional_data_value, dtype=tf.float32),
            tf.shape(img)[0], axis=0
        ), name="da_const"
    )(img_input)

    output = model([x_layer, additional_data_layer, img_input])
    return tf.keras.Model(inputs=img_input, outputs=output)

def generate_shap_map(submodel, image, class_idx, explainer, num_classes):
    """
    Computes SHAP values with GradientExplainer over the image sub-model.

    Returns
    -------
    importance_map : (H, W) float32 — sum of |SHAP| across channels
    sign_map       : (H, W) float32 — signed sum of SHAP
                     >0 → pixel pushes towards the positive class
                     <0 → pixel pushes away from the positive class
    """
    sample      = image[np.newaxis].astype(np.float32)
    shap_values = explainer.shap_values(sample)

    print(f"  [SHAP] type={type(shap_values)}", end="")

    # ── Extract array according to layout ──────────────────────────────────
    if isinstance(shap_values, list):
        if num_classes == 2:
            sv = np.array(shap_values[0])
        else:
            idx = class_idx if class_idx < len(shap_values) else 0
            sv  = np.array(shap_values[idx])
        print(f", len={len(shap_values)}, sv.shape={sv.shape}")
    else:
        sv = np.array(shap_values)
        print(f", sv.shape={sv.shape}")

    # ── Normalize to (n_samples, H, W, C) ────────────────────────────────────
    # GradientExplainer can return different layouts depending on the TF/shap
    # version:
    #   (a) list of arrays, each (n_samples, H, W, C)  ← already extracted
    #   (b) array (n_samples, H, W, C)                 ← direct 4D
    #   (c) array (n_outputs, n_samples, H, W, C)      ← 5D, first dim
    #   (d) array (n_samples, H, W, C, n_outputs)      ← 5D, last dim
    if sv.ndim == 5:
        if sv.shape[-1] <= 4:          # small C → outputs on dim -1
            idx = class_idx if sv.shape[-1] > class_idx else 0
            sv  = sv[..., idx]
        else:                          # outputs on dim 0
            idx = class_idx if sv.shape[0] > class_idx else 0
            sv  = sv[idx]

    while sv.ndim < 4:
        sv = sv[np.newaxis]

    sv_sample = sv[0]                  # → (H, W, C)
    print(f"  normalized sv_sample.shape: {sv_sample.shape}")

    if sv_sample.ndim == 2:
        return np.abs(sv_sample).astype(np.float32), sv_sample.astype(np.float32)

    if sv_sample.ndim != 3:
        raise RuntimeError(
            f"Unexpected shape: {sv_sample.shape} (original: {sv.shape}). "
            "(H, W, C) was expected."
        )

    importance_map = np.sum(np.abs(sv_sample), axis=-1).astype(np.float32)
    sign_map       = np.sum(sv_sample,          axis=-1).astype(np.float32)
    return importance_map, -sign_map

def freq_to_pixel(f, img_height, fmin, fmax):
    """Converts frequency to a vertical pixel (log scale, high frequency at the top)."""
    rel_pos = (np.log10(f) - np.log10(fmin)) / (np.log10(fmax) - np.log10(fmin))
    return int(img_height * (1 - rel_pos))

def calculate_band_importance(shap_resized, h, bands, f_min, f_max):
    """Returns dict band_name → percentage of SHAP importance."""
    total_saliency = np.sum(shap_resized) + 1e-9
    band_saliency  = {}
    for band_name, (f_low, f_high) in bands.items():
        y_bottom = freq_to_pixel(f_low,  h, f_min, f_max)
        y_top    = freq_to_pixel(f_high, h, f_min, f_max)
        y_top    = max(0, min(y_top, h - 1))
        y_bottom = max(0, min(y_bottom, h))
        band_sum = np.sum(shap_resized[y_top:y_bottom, :])
        band_saliency[band_name] = band_sum / total_saliency * 100.0
    return band_saliency

def calculate_band_ratio(shap_resized, h, w, bands, f_min, f_max):
    """Ratio = % SHAP / % band area (how much more or less than expected)."""
    total_saliency = np.sum(shap_resized) + 1e-9
    total_area     = h * w
    ratios         = {}
    for band_name, (f_low, f_high) in bands.items():
        y_bottom  = freq_to_pixel(f_low,  h, f_min, f_max)
        y_top     = freq_to_pixel(f_high, h, f_min, f_max)
        y_top     = max(0, min(y_top, h - 1))
        y_bottom  = max(0, min(y_bottom, h))
        band_area = max((y_bottom - y_top) * w, 1)
        shap_pct  = np.sum(shap_resized[y_top:y_bottom, :]) / total_saliency * 100.0
        area_pct  = band_area / total_area * 100.0
        ratios[band_name] = shap_pct / area_pct if area_pct > 0 else 0.0
    return ratios

def calculate_shap_entropy(shap_resized):
    """Shannon entropy of the SHAP map normalized as a distribution."""
    flat = shap_resized.flatten().astype(np.float64)
    flat = flat / (flat.sum() + 1e-12)
    flat = flat[flat > 0]
    return -np.sum(flat * np.log2(flat))

def calculate_band_center_of_mass(shap_resized, h, bands, f_min, f_max, n_freqs=100):
    """
    Returns:
        band_cm   : name of the band where the vertical SHAP center of mass falls
        freq_cm   : continuous frequency (Hz) of the center of mass
        freq_disc : closest discrete frequency out of n_freqs on a log scale (Hz)
        idx_disc  : (0-based) index of that discrete frequency
    """
    # Center of mass on the Y axis (pixel)
    ys, xs = np.indices(shap_resized.shape)
    total  = shap_resized.sum() + 1e-12
    cm_y   = np.sum(ys * shap_resized) / total

    # Convert pixel → continuous frequency (log scale, high frequency at the top)
    rel     = 1.0 - cm_y / h
    log_f   = np.log10(f_min) + rel * (np.log10(f_max) - np.log10(f_min))
    freq_cm = 10 ** log_f

    # Discrete frequencies on a log scale (the same ones used by the scalogram)
    freqs_log = np.logspace(np.log10(f_min), np.log10(f_max), n_freqs)
    idx_disc  = int(np.argmin(np.abs(freqs_log - freq_cm)))
    freq_disc = freqs_log[idx_disc]

    # Assign to a band
    band_cm = "out_of_range"
    for band_name, (f_low, f_high) in bands.items():
        if f_low <= freq_cm <= f_high:
            band_cm = band_name
            break

    return band_cm, freq_cm, freq_disc, idx_disc

def save_csv_row(csv_path, image_name, band_saliency, band_ratios,
                  entropy, band_cm, freq_cm, freq_disc, idx_disc, prob_pred, bands):
    """
    Appends (or creates) a row in the results CSV incrementally.
    Written after each image so progress is not lost.

    prob_pred : dict  e.g. {"prob_class1": 0.87}  (binary)
                           {"prob_class1": 0.1, "prob_class2": 0.9}  (multiclass)
    """
    columns = ["image"]
    for nb in bands:
        columns.append(f"pct_shap_{nb}")
    for nb in bands:
        columns.append(f"ratio_shap_{nb}")
    columns += ["shap_entropy", "center_of_mass_band",
                "freq_cm_hz", "discrete_freq_hz", "discrete_freq_idx"]
    columns += list(prob_pred.keys())

    row = {"image": image_name}
    for nb in bands:
        row[f"pct_shap_{nb}"]   = round(band_saliency.get(nb, 0.0), 4)
        row[f"ratio_shap_{nb}"] = round(band_ratios.get(nb, 0.0),   4)
    row["shap_entropy"]         = round(entropy,   4)
    row["center_of_mass_band"]  = band_cm
    row["freq_cm_hz"]           = round(float(freq_cm),   4)
    row["discrete_freq_hz"]     = round(float(freq_disc), 4)
    row["discrete_freq_idx"]    = int(idx_disc)
    row.update(prob_pred)

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        existing_df = existing_df[existing_df["image"] != image_name]
        new_df      = pd.concat([existing_df, pd.DataFrame([row])], ignore_index=True)
    else:
        new_df = pd.DataFrame([row], columns=columns)

    new_df.to_csv(csv_path, index=False)

def save_3panel_figure(
    original_image,
    importance_map,
    sign_map,
    band_saliency,
    file_name,
    output_path,
    bands,
    f_min, f_max,
    alpha=0.55,
    cmap_shap='hot',
    dpi=150,
    original_img_path=None,
):
    """
    Generates a 3-panel figure:
        [left]   Original scalogram loaded from disk (native jet, unprocessed)
        [center] SHAP importance |·| overlaid on a grayscale version
        [right]  SHAP sign (RdBu_r) overlaid on a grayscale version

    Colorbars are placed below each panel to avoid overlapping the images.
    Label-free version, with thicker lines and larger numbers.
    """
    # ── Panel 0: original scalogram from disk ─────────────────────────────
    if original_img_path and os.path.exists(original_img_path):
        img_orig_bgr = cv2.imread(original_img_path)
        img_panel0   = cv2.cvtColor(img_orig_bgr, cv2.COLOR_BGR2RGB)
        img_panel0   = cv2.resize(img_panel0, (224, 224), interpolation=cv2.INTER_AREA)
    else:
        img_vis = original_image.copy().astype(np.float32)
        img_vis = img_vis - img_vis.min()
        mx = img_vis.max()
        if mx > 1e-6:
            img_vis = img_vis / mx
        img_panel0 = (img_vis * 255).astype(np.uint8)

    h, w = img_panel0.shape[:2]

    # ── Grayscale base for heatmap panels ─────────────────────────────────
    img_gray = cv2.cvtColor(img_panel0, cv2.COLOR_RGB2GRAY)
    img_base = cv2.cvtColor(img_gray,   cv2.COLOR_GRAY2RGB)   # (H, W, 3) uint8

    # ── Normalize importance map → [0, 1] ─────────────────────────────────
    shap_resized = cv2.resize(importance_map, (w, h), interpolation=cv2.INTER_LINEAR)
    shap_norm    = (shap_resized - shap_resized.min()) / (
        shap_resized.max() - shap_resized.min() + 1e-8
    )

    # ── Normalize sign map centered at 0 ───────────────────────────────────
    sign_resized = cv2.resize(sign_map, (w, h), interpolation=cv2.INTER_LINEAR)
    vabs         = float(np.abs(sign_resized).max())
    vabs         = vabs if vabs > 1e-10 else 1.0
    sign_norm    = (sign_resized + vabs) / (2.0 * vabs)

    no_shap_mask = shap_norm < 1e-6

    # ── RGBA importance ────────────────────────────────────────────────────
    cmap_obj  = plt.get_cmap(cmap_shap)
    shap_rgba = cmap_obj(shap_norm).astype(np.float64)
    shap_rgba[..., 3] = np.where(no_shap_mask, 0.0, alpha)

    # ── RGBA sign ──────────────────────────────────────────────────────────
    cmap_sign = plt.get_cmap("RdBu_r")
    sign_rgba = cmap_sign(sign_norm).astype(np.float64)
    sign_rgba[..., 3] = np.where(no_shap_mask, 0.0, alpha)

    # ── Layout ──────────────────────────────────────────────────────────────
    col_w_in = 5.0
    img_h_in = col_w_in * (h / w)
    fig_w    = col_w_in * 5 + 2.5
    fig_h    = img_h_in + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        1, 3,
        left=0.02, right=0.72,
        bottom=0.14, top=0.90,
        wspace=0.06,
    )
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    # Time ticks on the X axis (0, 5, 10, 15, 20 s)
    duration_s = 20
    tick_s     = [0, 5, 10, 15, 20]
    tick_px    = [t / duration_s * (w - 1) for t in tick_s]

    def add_time_axis(ax):
        """
        Configures the axis to show only the X time axis (no text labels).
        """
        ax.set_xlim(0, w - 1)
        ax.set_ylim(h - 1, 0)
        ax.yaxis.set_visible(False)
        for spine in ('top', 'left', 'right'):
            ax.spines[spine].set_visible(False)
        ax.xaxis.set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['bottom'].set_color('black')
        ax.set_xticks(tick_px)
        # Text labels are left empty to remove the letters/s
        ax.set_xticklabels(["" for t in tick_s], fontsize=18, color='black')
        ax.tick_params(axis='x', which='both', bottom=True, top=False,
                       length=6, width=1.2, pad=4, colors='black')

    # ── Panel 0: original scalogram (native jet) ───────────────────────────
    ax0.imshow(img_panel0, aspect="auto", extent=[0, w - 1, h - 1, 0])
    add_time_axis(ax0)

    # ── Panel 1: importance over grayscale ─────────────────────────────────
    ax1.imshow(img_base,  aspect="auto", extent=[0, w - 1, h - 1, 0])
    ax1.imshow(shap_rgba, aspect="auto", extent=[0, w - 1, h - 1, 0])
    add_time_axis(ax1)
    sm_shap = mcm.ScalarMappable(cmap=cmap_obj, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm_shap.set_array([])

    # ── Panel 2: sign over grayscale ────────────────────────────────────────
    ax2.imshow(img_base,  aspect="auto", extent=[0, w - 1, h - 1, 0])
    ax2.imshow(sign_rgba, aspect="auto", extent=[0, w - 1, h - 1, 0])
    add_time_axis(ax2)
    sm_sign = mcm.ScalarMappable(
        cmap=cmap_sign, norm=mcolors.Normalize(vmin=-vabs, vmax=vabs)
    )
    sm_sign.set_array([])

    # ── Thicker band lines (without band-name text) ────────────────────────
    for band_name, (f_low, f_high) in bands.items():
        y_bottom = freq_to_pixel(f_low,  h, f_min, f_max)
        y_top    = freq_to_pixel(f_high, h, f_min, f_max)
        y_top    = max(0, min(y_top, h - 1))
        y_bottom = max(0, min(y_bottom, h))

        for ax in (ax0, ax1, ax2):
            # Thicker dashed lines (linewidth=2.5)
            ax.axhline(y_top, color='white', linestyle='--', linewidth=1.5, alpha=0.9)

    # ── Horizontal colorbars below each panel (no numbers or labels) ───────
    sm_jet = mcm.ScalarMappable(cmap=plt.get_cmap('jet'),
                                norm=mcolors.Normalize(vmin=0, vmax=1))
    sm_jet.set_array([])
    pos0  = ax0.get_position()
    cax0  = fig.add_axes([pos0.x0, 0.04, pos0.width, 0.03])
    cb0   = fig.colorbar(sm_jet, cax=cax0, orientation='horizontal')
    cb0.ax.set_xticklabels([])  # Remove the numbers

    pos1  = ax1.get_position()
    cax1  = fig.add_axes([pos1.x0, 0.04, pos1.width, 0.03])
    cb1   = fig.colorbar(sm_shap, cax=cax1, orientation='horizontal')
    cb1.ax.set_xticklabels([])  # Remove the numbers

    pos2c = ax2.get_position()
    cax2  = fig.add_axes([pos2c.x0, 0.04, pos2c.width, 0.03])
    cb2   = fig.colorbar(sm_sign, cax=cax2, orientation='horizontal', format='%.2f')
    cb2.ax.set_xticklabels([])  # Remove the numbers

    # ── Save ─────────────────────────────────────────────────────────────────
    svg_file_name = file_name.replace('.png', '.svg').replace('.jpg', '.svg')

    fig_path     = os.path.join(output_path, svg_file_name)
    fig_png_path = os.path.join(output_path, file_name)

    plt.savefig(fig_path, format='svg', bbox_inches='tight')
    plt.savefig(fig_png_path, format='png', bbox_inches='tight')
    plt.close(fig)

    return fig_path

def maps_shap(
    data_path,
    model_path,
    window_size,
    step_size,
    output_path,
    train_data_path,
    use_all=False,
    batch_size=32,
    n_background=50,
):
    """
    Complete SHAP pipeline for scalograms.

    Parameters
    ----------
    data_path        : Validation directory (subfolders per class).
    model_path       : Path to the .hdf5 / .keras model.
    window_size      : Time-signal window size.
    step_size        : Window step.
    output_path      : Root output directory.
    train_data_path  : Training directory (for the SHAP background).
    use_all          : True  → process all samples of each class.
                       False → only the correctly classified ones.
    batch_size       : Batch size for the global prediction.
    n_background     : Number of background scalograms for GradientExplainer.
    """

    # ── 1. Model ──
    model        = load_trained_model(model_path)
    preprocessor = model_preprocessors[backbone]

    # ── 2. Validation data ──
    print("Loading validation data...")
    x, x_img, y_true, additional_data, num_classes, classes, keys, img_paths = preprocess_data(
        data_path, window_size, step_size,
        scales=None, master_path=master_file,
        additional_columns=additional_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        img_preprocessor=preprocessor,
    )

    # ── 3. Background from training data ──
    print("Loading training background...")
    _, x_img_train, _, _, _, _, _, _ = preprocess_data(
        train_data_path, window_size, step_size,
        scales=None, master_path=master_file,
        additional_columns=additional_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        img_preprocessor=preprocessor,
    )

    bg_idx = np.random.choice(
        len(x_img_train),
        size=min(n_background, len(x_img_train)),
        replace=False,
    )
    background_imgs_global = x_img_train[bg_idx].astype(np.float32)
    print(f"Background ready: {background_imgs_global.shape}")

    # ── 4. Global predictions ──
    y_prob = model.predict([x, additional_data, x_img], batch_size=batch_size)
    y_pred_labels = (
        (y_prob > 0.5).astype(int).flatten()
        if num_classes == 2
        else y_prob.argmax(axis=1)
    )

    # ── 5. Output folders ──
    output_folder = os.path.join(output_path, "SHAP_Maps")
    os.makedirs(output_folder, exist_ok=True)
    for i in range(num_classes):
        os.makedirs(os.path.join(output_folder, f"G{i+1}"), exist_ok=True)

    # ── 6. Loop over classes ──────────────────────────────────────────────────
    for class_idx in range(num_classes):
        print(f"\n{'='*60}")
        print(f"  Class {class_idx + 1} / {num_classes}")
        print(f"{'='*60}")

        if use_all:
            class_indices = [i for i in range(len(y_true)) if y_true[i] == class_idx]
        else:
            class_indices = [
                i for i in range(len(y_true))
                if y_true[i] == class_idx and y_true[i] == y_pred_labels[i]
            ]

        if not class_indices:
            print("  No samples. Skipping.")
            continue

        x_small          = x[class_indices]
        x_img_small      = x_img[class_indices]
        additional_data_small = additional_data[class_indices]
        keys_small       = [keys[i]      for i in class_indices]
        paths_small      = [img_paths[i] for i in class_indices]
        y_prob_small     = y_prob[class_indices]   # model probabilities per sample

        output_dir = os.path.join(output_folder, f"G{class_idx+1}")

        # Results CSV for this class (updated after each image)
        class_csv_path = os.path.join(output_dir, f"shap_metrics_G{class_idx+1}.csv")

        # ── 7. Loop over images ─────────────────────────────────────────────
        for i in tqdm(range(len(x_small)), desc=f"Class {class_idx+1}"):

            key_str    = keys_small[i] if i < len(keys_small) else str(i)
            fig_name   = f"{key_str}_{i}_shap_3panels.png"
            img_path_i = paths_small[i]

            # Sub-model with x and additional_data fixed for this sample
            submodel = build_image_submodel(model, x_small[i], additional_data_small[i])

            assert background_imgs_global.shape[0] > 0, \
                "background_imgs_global is empty — check train_data_path"
            assert background_imgs_global.dtype == np.float32, \
                f"unexpected dtype: {background_imgs_global.dtype}"

            explainer = shap.GradientExplainer(
                submodel,
                background_imgs_global,
            )

            importance_map, sign_map = generate_shap_map(
                submodel, x_img_small[i], class_idx, explainer, num_classes
            )

            # ── Importance per band (on the preprocessed 224×224 image) ─────
            h, w = x_img_small[i].shape[:2]
            shap_resized = cv2.resize(
                importance_map, (w, h), interpolation=cv2.INTER_LINEAR
            )
            shap_resized = (shap_resized - shap_resized.min()) / (
                shap_resized.max() - shap_resized.min() + 1e-8
            )

            band_saliency = calculate_band_importance(
                shap_resized, h, bands, f_min, f_max
            )
            band_ratios   = calculate_band_ratio(
                shap_resized, h, w, bands, f_min, f_max
            )
            entropy       = calculate_shap_entropy(shap_resized)
            band_cm, freq_cm, freq_disc, idx_disc = calculate_band_center_of_mass(
                shap_resized, h, bands, f_min, f_max, n_freqs=100
            )

            # ── 3-panel figure ────────────────────────────────────────────────
            save_3panel_figure(
                original_image     = x_img_small[i],
                importance_map      = importance_map,
                sign_map            = sign_map,
                band_saliency       = band_saliency,
                file_name           = fig_name,
                output_path         = output_dir,
                bands               = bands,
                f_min               = f_min,
                f_max               = f_max,
                alpha               = 0.55,
                cmap_shap           = 'hot',
                original_img_path   = img_path_i,
            )

            # ── Probability predicted by the model ────────────────────────────
            prob_raw = y_prob_small[i].flatten()
            # Binary: probability of class 1. Multiclass: probability of each class.
            if num_classes == 2:
                prob_pred = {"prob_class1": round(float(prob_raw[0]), 6)}
            else:
                prob_pred = {f"prob_class{c+1}": round(float(prob_raw[c]), 6)
                             for c in range(num_classes)}

            # ── Save row in CSV (incremental) ──────────────────────────────────
            save_csv_row(
                csv_path      = class_csv_path,
                image_name    = fig_name,
                band_saliency = band_saliency,
                band_ratios   = band_ratios,
                entropy       = entropy,
                band_cm       = band_cm,
                freq_cm       = freq_cm,
                freq_disc     = freq_disc,
                idx_disc      = idx_disc,
                prob_pred     = prob_pred,
                bands         = bands,
            )

            max_band = max(band_saliency, key=band_saliency.get)
            print(
                f"  img {i} [{key_str}] → dominant band: {max_band} "
                f"({band_saliency[max_band]:.1f}%) | "
                f"CM: {band_cm} {freq_cm:.2f} Hz "
                f"(f{idx_disc+1} = {freq_disc:.2f} Hz) | "
                f"H(SHAP): {entropy:.2f} bits"
            )

            # Free the sub-model so TF graphs do not accumulate in memory
            del submodel, explainer
            gc.collect()

    print("\nSHAP maps saved to:", output_folder)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    maps_shap(
        data_path        = data_path,
        model_path       = model_path,
        window_size      = window_size,
        step_size        = step_size,
        output_path      = output_path,
        train_data_path  = train_data_path,
        use_all          = True,   # True → all samples, False → only correct ones
        batch_size       = 32
    )
