# Learnable FFT Layer for Biosignal Classification

We propose a novel, learnable Fast Fourier Transform (FFT) layer designed specifically for advanced biosignal analysis. Traditional deep learning models often exhibit a strong bias towards low frequencies when processing raw time-series data, potentially ignoring subtle high-frequency biomarkers. This architecture mitigates that bias by introducing adaptive frequency targeting and self-attention, significantly improving downstream classification performance.

## Core Mechanism: The Learnable FFT Module

The custom layer transforms raw biological signals into a highly concentrated, attention-weighted spectral embedding. It operates through three main stages:

* **Dual Spectral Representation:** The module computes a standard real FFT (fixed up to 40 Hz) and pairs it with a **Learnable Fourier Basis**. By adding trainable weights to base frequencies ($f_{base} + \Delta f$), the network dynamically adjusts the exact frequencies it needs to monitor during training. These two representations are concatenated into a unified spectrum.
* **Frequency-Frequency Attention:** To determine which frequencies matter most, the fused spectrum passes through a self-attention mechanism. Using 1D Convolutions, it projects the data into Query, Key, and Value vectors, calculating a softmax attention map that highlights critical frequency combinations while suppressing noise. A residual connection preserves the original normalized input.
* **Targeted Feature Pooling:** Instead of outputting the entire high-dimensional spectrum, the layer compresses the attention-weighted signal into a dense **18-dimensional embedding** per channel. This vector explicitly captures:
  * Global spectral statistics (Mean, Standard Deviation, Total Energy).
  * The magnitudes and exact frequencies of the top 5 highest peaks.
  * Average power across 5 clinically relevant physiological subbands (e.g., Delta, Theta, Alpha, Beta, Gamma).

## Full Multimodal Architecture

The Learnable FFT layer is designed to function as the primary spectral feature extractor within a larger multimodal network. The full architecture leverages Transformers and Cross-Attention to fuse temporal, spectral, visual, and clinical data.

* **bERG Signal Branch:** Processes raw time-series data through parallel streams. The temporal stream uses a TimeDistributed 1D CNN, while the spectral stream uses our TimeDistributed Learnable FFT module. Both pass through Transformer Encoders and are integrated using Temporal ↔ Spectral Cross-Attention.
* **Scalogram Branch:** Extracts visual time-frequency features from scalogram images using a VGG16 backbone followed by a Vision Transformer. This representation is fused with the processed bERG signal via Signal ↔ Scalogram Cross-Attention.
* **Clinical Branch:** Feeds numerical clinical variables through a Multi-Layer Perceptron (MLP). A final Multimodal ↔ Clinical Cross-Attention layer merges the physiological/visual features with the patient's clinical profile.
* **Classifier:** The fully fused representation passes through dense layers to output the final binary classification.

## Implementation

The custom layer is built using TensorFlow and Keras. You can find the complete implementation in the **`classes.py`** script under the class name **`FFT_layer`**.

Furthermore, we complement the FFT branch with a temporal extraction module using 1D Convolutional Neural Networks, an image feature extraction module designed for scalograms via a VGG16 backbone, and a clinical data module to enhance the classification of basal ERG (bERG) signals. The complete architecture can be found in the `model.py` script under the name `multimodal_erg_model`.

<img width="716" height="796" alt="Imagen1" src="https://github.com/user-attachments/assets/3c7f8a5e-931a-4c30-89b3-403991edb608" />


