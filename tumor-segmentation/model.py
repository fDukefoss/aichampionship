import tensorflow as tf
from tensorflow.keras import layers, models

def build_multiscale_anatomy_fcn(input_shape=(None, None, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    # Scale 1 - Full resolution
    x1 = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x1 = layers.MaxPooling2D(2)(x1)
    x1 = layers.Conv2D(64, 3, padding='same', activation='relu')(x1)
    x1 = layers.MaxPooling2D(2)(x1)

    # Scale 2 - Downsampled input
    x2 = layers.AveragePooling2D(pool_size=2)(inputs)
    x2 = layers.Conv2D(32, 3, padding='same', activation='relu')(x2)
    x2 = layers.MaxPooling2D(2)(x2)
    x2 = layers.Conv2D(64, 3, padding='same', activation='relu')(x2)
    x2 = layers.UpSampling2D(size=2)(x2)

    # Resize x2 to match x1 dynamically
    def resize_to_x1(x):
        x2, x1 = x
        target_shape = tf.shape(x1)[1:3]
        return tf.image.resize(x2, target_shape, method='bilinear')

    x2_resized = layers.Lambda(resize_to_x1)([x2, x1])

    # Merge paths
    x = layers.Concatenate()([x1, x2_resized])
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=2)(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=2)(x)
    output = layers.Conv2D(1, 1, activation='sigmoid')(x)

    return models.Model(inputs=inputs, outputs=output)
