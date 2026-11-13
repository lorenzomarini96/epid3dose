
"""
3D U-Net with Early Fusion and Weighted EPID Lifting.

Inputs:
    CT:   (128, 128, 128, 1)
    EPID: (256, 256, 1)

Output:
    Reconstructed dose volume: (128, 128, 128, 1)

Architecture:
    - Early fusion of CT and EPID
    - Bilinear EPID resizing
    - Weighted EPID lifting along Y
    - 3D U-Net encoder-decoder
    - Softplus output activation
"""

import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Conv3D,
    MaxPooling3D,
    Conv3DTranspose,
    BatchNormalization,
    LeakyReLU,
    Concatenate,
    Lambda,
    Resizing,
)

from tensorflow.keras.models import Model


# =============================================================================
# 3D CONVOLUTIONAL BLOCK
# =============================================================================

def conv_block_3d(x, f, a=0.1, name=None):
    """Two consecutive Conv3D + BatchNormalization + LeakyReLU layers."""

    x = Conv3D(
        f,
        3,
        padding="same",
        use_bias=False,
        name=None if name is None else f"{name}_conv1"
    )(x)

    x = BatchNormalization(
        name=None if name is None else f"{name}_bn1"
    )(x)

    x = LeakyReLU(
        a,
        name=None if name is None else f"{name}_lrelu1"
    )(x)

    x = Conv3D(
        f,
        3,
        padding="same",
        use_bias=False,
        name=None if name is None else f"{name}_conv2"
    )(x)

    x = BatchNormalization(
        name=None if name is None else f"{name}_bn2"
    )(x)

    x = LeakyReLU(
        a,
        name=None if name is None else f"{name}_lrelu2"
    )(x)

    return x


# =============================================================================
# WEIGHTED EPID LIFTING
# =============================================================================

def weighted_epid2d_to_epid3d(
    epid2d,
    y_size=128,
    alpha=1.0,
    name="epid_lift_weightedY"
):
    """
    Lift a 2D EPID image into a 3D volume.

    EPID is interpreted in the (Z, X) plane:

        Input:  (B, Z, X, 1)
        Output: (B, Z, Y, X, 1)

    The image is replicated along Y and weighted using an
    exponentially decreasing function:

        w(y) = exp(-alpha * y)

    The highest weight is assigned to y = 0.
    """

    def _lift(e):

        # Expand Y dimension.
        e = tf.expand_dims(e, axis=2)

        # Replicate along Y.
        e = tf.tile(
            e,
            [1, 1, y_size, 1, 1]
        )

        # Normalized Y coordinate.
        y = tf.linspace(
            0.0,
            1.0,
            y_size
        )

        # Exponentially decreasing weights.
        w = tf.exp(-alpha * y)

        # Reshape for broadcasting.
        w = tf.reshape(
            w,
            [1, 1, y_size, 1, 1]
        )

        return e * w

    return Lambda(
        _lift,
        name=name
    )(epid2d)


# =============================================================================
# MODEL
# =============================================================================

def build_model(
    base_filters=32,
    alpha=0.1,
    D=128,
    lift_alpha=1.0
):
    """
    Build the 3D U-Net with Early Fusion and Weighted EPID Lifting.

    Parameters
    ----------
    base_filters : int
        Number of filters in the first encoder block.

    alpha : float
        Negative slope of LeakyReLU.

    D : int
        Spatial dimension of the 3D input volume.

    lift_alpha : float
        Exponential decay parameter used in EPID lifting.

    Returns
    -------
    model : tensorflow.keras.Model
        3D U-Net model.
    """

    # -----------------------------------------------------------------
    # INPUTS
    # -----------------------------------------------------------------

    ct_in = Input(
        (D, D, D, 1),
        name="ct"
    )

    ep_in = Input(
        (256, 256, 1),
        name="epid"
    )

    # -----------------------------------------------------------------
    # EPID RESIZING: 256 x 256 -> 128 x 128
    # -----------------------------------------------------------------

    ep = Resizing(
        height=D,
        width=D,
        interpolation="bilinear",
        name="epid_resize"
    )(ep_in)

    # -----------------------------------------------------------------
    # WEIGHTED EPID LIFTING
    # -----------------------------------------------------------------

    ep3 = weighted_epid2d_to_epid3d(
        ep,
        y_size=D,
        alpha=lift_alpha,
        name="epid_lift_weightedY"
    )

    # -----------------------------------------------------------------
    # EARLY FUSION
    # -----------------------------------------------------------------

    x = Concatenate(
        axis=-1,
        name="early_concat"
    )([ct_in, ep3])

    # -----------------------------------------------------------------
    # ENCODER
    # -----------------------------------------------------------------

    c1 = conv_block_3d(
        x,
        base_filters,
        alpha,
        name="enc1"
    )

    p1 = MaxPooling3D(
        pool_size=2,
        name="pool1"
    )(c1)

    c2 = conv_block_3d(
        p1,
        base_filters * 2,
        alpha,
        name="enc2"
    )

    p2 = MaxPooling3D(
        pool_size=2,
        name="pool2"
    )(c2)

    c3 = conv_block_3d(
        p2,
        base_filters * 4,
        alpha,
        name="enc3"
    )

    p3 = MaxPooling3D(
        pool_size=2,
        name="pool3"
    )(c3)

    # -----------------------------------------------------------------
    # BOTTLENECK
    # -----------------------------------------------------------------

    b = conv_block_3d(
        p3,
        base_filters * 8,
        alpha,
        name="bottleneck"
    )

    # -----------------------------------------------------------------
    # DECODER
    # -----------------------------------------------------------------

    u3 = Conv3DTranspose(
        base_filters * 4,
        2,
        strides=2,
        padding="same",
        name="up3"
    )(b)

    u3 = Concatenate(
        name="skip3"
    )([u3, c3])

    d3 = conv_block_3d(
        u3,
        base_filters * 4,
        alpha,
        name="dec3"
    )

    u2 = Conv3DTranspose(
        base_filters * 2,
        2,
        strides=2,
        padding="same",
        name="up2"
    )(d3)

    u2 = Concatenate(
        name="skip2"
    )([u2, c2])

    d2 = conv_block_3d(
        u2,
        base_filters * 2,
        alpha,
        name="dec2"
    )

    u1 = Conv3DTranspose(
        base_filters,
        2,
        strides=2,
        padding="same",
        name="up1"
    )(d2)

    u1 = Concatenate(
        name="skip1"
    )([u1, c1])

    d1 = conv_block_3d(
        u1,
        base_filters,
        alpha,
        name="dec1"
    )

    # -----------------------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------------------

    out = Conv3D(
        1,
        1,
        activation="softplus",
        name="dose"
    )(d1)

    # -----------------------------------------------------------------
    # MODEL DEFINITION
    # -----------------------------------------------------------------

    return Model(
        inputs=[ct_in, ep_in],
        outputs=out,
        name="unet_3d"
    )
