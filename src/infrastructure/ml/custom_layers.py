"""Custom Keras layers for spoofing detection model."""
import keras
from keras import layers
import tensorflow as tf


@keras.saving.register_keras_serializable()
class SpecAugment(keras.layers.Layer):
    """
    SpecAugment layer for spectral augmentation.

    During training, applies time and frequency masking to spectrograms.
    During inference, passes input through unchanged.

    This layer is needed to load models that were trained with SpecAugment.
    """

    def __init__(self, time_mask_param=40, freq_mask_param=8, **kwargs):
        """
        Initialize SpecAugment layer.

        Args:
            time_mask_param: Maximum time mask width
            freq_mask_param: Maximum frequency mask width
        """
        super().__init__(**kwargs)
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def call(self, x, training=False):
        """
        Forward pass.

        During inference (training=False), returns input unchanged.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Input tensor (unchanged during inference)
        """
        # No augmentation during inference
        return x

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'time_mask_param': self.time_mask_param,
            'freq_mask_param': self.freq_mask_param
        })
        return config


@keras.saving.register_keras_serializable()
class SEBlock(keras.layers.Layer):
    """
    Squeeze-and-Excitation block for channel attention.

    This implements the SE mechanism to adaptively recalibrate
    channel-wise feature responses by explicitly modeling
    interdependencies between channels.

    Reference: Hu et al., "Squeeze-and-Excitation Networks"
    """

    def __init__(self, ratio=8, **kwargs):
        """
        Initialize SE block.

        Args:
            ratio: Reduction ratio for squeeze operation
        """
        super().__init__(**kwargs)
        self.ratio = ratio
        self.channels = None
        self.fc1 = None
        self.fc2 = None

    def build(self, input_shape):
        """
        Build layer weights.

        Args:
            input_shape: Shape of input tensor
        """
        self.channels = input_shape[-1]
        self.fc1 = layers.Dense(self.channels // self.ratio, activation='relu')
        self.fc2 = layers.Dense(self.channels, activation='sigmoid')
        super().build(input_shape)

    def call(self, x):
        """
        Forward pass.

        Applies squeeze-and-excitation mechanism:
        1. Squeeze: Global average pooling
        2. Excitation: Two FC layers with relu and sigmoid
        3. Scale: Multiply input by excitation weights

        Args:
            x: Input tensor (2D or 3D)

        Returns:
            Recalibrated tensor with same shape as input
        """
        # Use static shape to determine input dimensionality
        input_rank = len(x.shape)

        if input_rank == 2:
            # 2D input: x shape is (batch, features)
            # No squeeze needed, apply FC layers directly
            excitation = self.fc1(x)
            excitation = self.fc2(excitation)
            return x * excitation
        else:
            # 3D input: x shape is (batch, time, features)
            # Squeeze: global average pooling across time dimension
            squeeze = tf.reduce_mean(x, axis=1, keepdims=True)
            # Excitation
            excitation = self.fc1(squeeze)
            excitation = self.fc2(excitation)
            return x * excitation

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({'ratio': self.ratio})
        return config
