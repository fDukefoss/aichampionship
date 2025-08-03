import tensorflow as tf
import os

def load_image_pair(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask > 200, tf.float32)  # Binarize
    return image, mask


def get_train_val_datasets(data_dir='data/patients', batch_size=1, val_split=0.2):
    image_dir = os.path.join(data_dir, 'imgs')
    mask_dir = os.path.join(data_dir, 'labels')

    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

    dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
    dataset = dataset.shuffle(len(image_files), seed=42)

    split_index = int(len(dataset) * (1 - val_split))
    train_ds = dataset.take(split_index)
    val_ds = dataset.skip(split_index)

    train_ds = train_ds.map(lambda x, y: load_image_pair(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: load_image_pair(x, y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
