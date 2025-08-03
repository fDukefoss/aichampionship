import tensorflow as tf
import numpy as np
from model import build_multiscale_anatomy_fcn  # from earlier
from loader import get_train_val_datasets  # to be implemented

# Dice coefficient
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# Loss
def dice_loss(y_true, y_pred): 
    return 1.0 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred): 
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# Training loop
def train(model, train_ds, val_ds, epochs=20, log_dir="logs/anatomy_fcn"):
    optimizer = tf.keras.optimizers.Adam()
    writer = tf.summary.create_file_writer(log_dir)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        for step, (x, y) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                pred = model(x, training=True)
                y = tf.image.resize_with_crop_or_pad(y, tf.shape(pred)[1], tf.shape(pred)[2])

                loss = combined_loss(y, pred)
                dice = dice_coef(y, pred)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Validation
        val_loss, val_dice = 0.0, 0.0
        for x_val, y_val in val_ds:
            pred_val = model(x_val, training=False)
            val_loss += combined_loss(y_val, pred_val).numpy()
            val_dice += dice_coef(y_val, pred_val).numpy()

        val_loss /= len(val_ds)
        val_dice /= len(val_ds)

        with writer.as_default():
            tf.summary.scalar("Val/Loss", val_loss, step=epoch)
            tf.summary.scalar("Val/Dice", val_dice, step=epoch)
            tf.summary.image("Val/Input", x_val, step=epoch, max_outputs=1)
            tf.summary.image("Val/Prediction", pred_val, step=epoch, max_outputs=1)
            tf.summary.image("Val/GT_Mask", y_val, step=epoch, max_outputs=1)

    model.save(f"{log_dir}/model_final.h5")
    print("Training complete.")

if __name__ == "__main__":
    model = build_multiscale_anatomy_fcn()
    model.compile(optimizer='adam', loss=combined_loss)

    train_ds, val_ds = get_train_val_datasets() 
    train(model, train_ds, val_ds, epochs=25)
