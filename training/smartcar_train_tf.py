import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf

from config.classes import SMARTCAR_CLASSES
from models.cnn_tf import create_smartcar_cnn


data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.RandomContrast(0.1),
    ]
)


def preprocess_train(x, y):
    x = tf.cast(x, tf.float32) / 127.5 - 1.0
    x = data_augmentation(x, training=True)
    return x, tf.argmax(y, axis=1)


def preprocess_eval(x, y):
    x = tf.cast(x, tf.float32) / 127.5 - 1.0
    return x, tf.argmax(y, axis=1)


def get_datasets(batch_size=8, img_size=96):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "data/smartcar/train",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
    )
    class_names = train_ds.class_names
    idx_to_class = {i: name for i, name in enumerate(class_names)}

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "data/smartcar/val",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        "data/smartcar/test",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
    )

    train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_eval, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_eval, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, idx_to_class


def train(epochs=100):
    train_ds, val_ds, test_ds, idx_to_class = get_datasets()

    model = create_smartcar_cnn(num_classes=len(SMARTCAR_CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    print(f"Classes: {idx_to_class}")
    print(
        f"Train batches: {len(train_ds)}, Val batches: {len(val_ds)}, Test batches: {len(test_ds)}"
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "smartcar_model_tf.h5",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=2,
    )

    model.save_weights("smartcar_model_tf.weights.h5")

    import json

    with open("idx_to_class_tf.json", "w") as f:
        json.dump(idx_to_class, f)
    print("Index to class mapping saved to idx_to_class_tf.json")


if __name__ == "__main__":
    train()
