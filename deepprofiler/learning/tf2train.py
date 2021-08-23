import comet_ml
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import random
tf.compat.v1.enable_v2_behavior()
tf.config.run_functions_eagerly(True)

AUTOTUNE = tf.data.AUTOTUNE


def make_dataset(path, batch_size, single_cell_metadata, config, is_training):
    @tf.function
    def fold_channels(crop):
        assert tf.executing_eagerly()

        crop = crop.numpy()
        output = np.reshape(crop, (crop.shape[0], crop.shape[0], -1), order="F").astype(np.float)
        #for i in range(output.shape[-1]):
        #    mean = np.mean(output[:, :, i])
        #    std = np.std(output[:, :, i])
        #    output[:, :, i] = (output[:, :, i] - mean) / std
        return tf.convert_to_tensor(output, dtype=tf.float32)

    def parse_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image, channels=1)
        image = tf.py_function(func=fold_channels, inp=[image], Tout=tf.float32)
        return image

    def configure_for_performance(ds, is_training):
        ds = ds.shuffle(buffer_size=20000)
        #if is_training:
        #    ds = augment(ds)

        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def random_illumination(image):
        if tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(0.5, tf.float32)):
        # Make channels independent images
            numchn = len(config["dataset"]["images"]["channels"])
            source = tf.transpose(image, [2, 1, 0])
            source = tf.expand_dims(source, -1)
            source = tf.image.grayscale_to_rgb(source)
        
            # Apply illumination augmentations
            bright = tf.random.uniform([numchn], minval=-0.2, maxval=0.2, dtype=tf.float32)
            channels = [tf.image.adjust_brightness(source[s,...], bright[s]) for s in range(numchn)]
            contrast = tf.random.uniform([numchn], minval=0.5, maxval=1.5, dtype=tf.float32)
            channels = [tf.image.adjust_contrast(channels[s], contrast[s]) for s in range(numchn)]
            result = tf.concat([tf.expand_dims(t, 0) for t in channels], axis=0)
    
            # Recover multi-channel image
            result = tf.image.rgb_to_grayscale(result)
            result = tf.transpose(result[:,:,:,0], [2, 1, 0])
        
            return result
        else:
            return image

    def random_crop_or_rotate(image):
        w, h, c = config["dataset"]["locations"]["box_size"], config["dataset"]["locations"]["box_size"], len(config["dataset"]["images"]["channels"])
        if tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(0.5, tf.float32)):
            size = tf.random.uniform([1], minval=int(w * 0.6), maxval=w, dtype=tf.int32)
            image = tf.image.random_crop(image, [size[0], size[0], c])
        else:
            image = tfa.image.rotate(image, tf.constant(np.pi / 9))
            image = tf.reshape(image, (w, h, c))
            image = tf.image.central_crop(image, 0.8)
        return tf.image.resize(image, (w, h))

    def augment(ds):
        ds = ds.map(
            lambda image, label: (random_crop_or_rotate(image), label), num_parallel_calls=AUTOTUNE
        ).map(
            lambda image, label: (tf.image.random_flip_left_right(image), label), num_parallel_calls=AUTOTUNE
        ).map(
            lambda image, label: (random_illumination(image), label), num_parallel_calls=AUTOTUNE
        ).map(
            lambda image, label: (tf.image.random_flip_up_down(image), label), num_parallel_calls=AUTOTUNE
        ).map(
            lambda image, label: (
            tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)), label),
            num_parallel_calls=AUTOTUNE
        )
        return ds

    filenames = [os.path.join(path, i) for i in single_cell_metadata["Image_Name"]]
    steps = np.math.ceil(len(filenames) / batch_size)
    random.shuffle(filenames)
    # print(labels)
    filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
    images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels = tf.keras.utils.to_categorical(single_cell_metadata["Categorical"])
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((images_ds, labels_ds))
    ds = configure_for_performance(ds, is_training)
    return ds, steps


def setup_callbacks(config, strategy_lr):
    callbacks = []
    
    # CSV Log
    csv_output = config["paths"]["logs"] + "/log.csv"
    callback_csv = tf.keras.callbacks.CSVLogger(filename=csv_output)
    callbacks.append(callback_csv)
    
    # Checkpoints
    output_file = config["paths"]["checkpoints"] + "/checkpoint_{epoch:04d}.hdf5"
    period = 1
    save_best = False
    if "checkpoint_policy" in config["train"]["model"] and isinstance(
            config["train"]["model"]["checkpoint_policy"], int):
        period = int(config["train"]["model"]["checkpoint_policy"])
    elif "checkpoint_policy" in config["train"]["model"] and config["train"]["model"]["checkpoint_policy"] == 'best':
        save_best = True

    callback_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_file,
        save_weights_only=True,
        save_best_only=save_best,
        period=period
    )
    callbacks.append(callback_model_checkpoint)
    epochs = config["train"]["model"]["epochs"]
    if "lr_schedule" in config["train"]["model"]:
        if config["train"]["model"]["lr_schedule"] == "cosine":
            lr_schedule_epochs = [x for x in range(epochs)]
            init_lr = config["train"]["model"]["params"]["learning_rate"]
            # Linear warm up
            lr_schedule_lr = [init_lr/(5-t) for t in range(5)]
            # Cosine decay
            lr_schedule_lr += [0.5 * (1 + np.cos((np.pi * t) / epochs)) * init_lr for t in range(5, epochs)]
        else:
            assert len(config["train"]["model"]["lr_schedule"]["epoch"]) == \
                   len(config["train"]["model"]["lr_schedule"]["lr"]), "Make sure that the length of " \
                                                                        "lr_schedule->epoch equals the length of " \
                                                                        "lr_schedule->lr in the config file."

            lr_schedule_epochs = config["train"]["model"]["lr_schedule"]["epoch"]
            lr_schedule_lr = config["train"]["model"]["lr_schedule"]["lr"]

        def lr_schedule(epoch, lr):
            if lr_schedule_epochs and epoch in lr_schedule_epochs:
                return lr_schedule_lr[lr_schedule_epochs.index(epoch)]
            else:
                return lr
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1))

    return callbacks


def setup_comet_ml(config):
    if 'comet_ml' in config["train"].keys():
        experiment = comet_ml.Experiment(
            api_key=config["train"]["comet_ml"]["api_key"],
            project_name=config["train"]["comet_ml"]["project_name"],
            auto_param_logging=True,
            auto_histogram_weight_logging=False,
            auto_histogram_gradient_logging=False,
            auto_histogram_activation_logging=False
        )
        if config["experiment_name"] != "results":
            experiment.set_name(config["experiment_name"])
        experiment.log_others(config)
    else:
        experiment = None
    return experiment


def learn_model(config, epoch):
    device_type = 'GPU'
    devices = tf.config.experimental.list_physical_devices(
        device_type)
    devices_names = [d.name.split("e:")[1] for d in devices]
    print(devices_names)
    strategy = tf.distribute.MirroredStrategy(devices=devices_names)

    BATCH_SIZE_PER_REPLICA = config["train"]["model"]["params"]["batch_size"]
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    strategy_lr = config["train"]["model"]["params"]["learning_rate"] * strategy.num_replicas_in_sync

    experiment = setup_comet_ml(config)
    single_cell_metadata = pd.read_csv(os.path.join(config["paths"]["single_cell_sample"], "expanded_sc_metadata.csv"))
    single_cell_metadata = single_cell_metadata[["Class_Name", "Image_Name", "Training_Status"]]
    single_cell_metadata = single_cell_metadata[single_cell_metadata["Training_Status"].isin(["SingleCellTraining", "SingleCellValidation"])]
    
    num_classes = len(pd.unique(single_cell_metadata["Class_Name"]))
    single_cell_metadata["Categorical"] = pd.Categorical(single_cell_metadata["Class_Name"]).codes

    path = config["paths"]["single_cell_sample"]
    dataset, steps_per_epoch = make_dataset(path, BATCH_SIZE,
                                            single_cell_metadata[single_cell_metadata["Training_Status"] == "SingleCellTraining"],
                                            config,
                                            is_training=True)
    validation_dataset, _ = make_dataset(path, BATCH_SIZE,
                                         single_cell_metadata[single_cell_metadata["Training_Status"] == "SingleCellValidation"],
                                         config,
                                         is_training=False)

    with strategy.scope():
        input_shape = (config["dataset"]["locations"]["box_size"], config["dataset"]["locations"]["box_size"],
                       len(config["dataset"]["images"]["channels"]))
        input_image = tf.keras.layers.Input(input_shape)

        model = tf.keras.applications.efficientnet.EfficientNetB0(
            include_top=False, weights=None, input_tensor=input_image,
            input_shape=input_shape
        )

        features = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(model.output)
        y = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(features)

        model = tf.keras.models.Model(inputs=input_image, outputs=y)

        regularizer = tf.keras.regularizers.l2(0.0001)
        for layer in model.layers:
            if hasattr(layer, "kernel_regularizer"):
                setattr(layer, "kernel_regularizer", regularizer)

        model = tf.keras.models.model_from_json(model.to_json())
        optimizer = tf.keras.optimizers.SGD(learning_rate = strategy_lr)
        loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        model.compile(optimizer, loss_func,
                     metrics=["accuracy", tfa.metrics.F1Score(num_classes=num_classes, average='macro')])

    callbacks = setup_callbacks(config, strategy_lr)

    if epoch == 1 and config["train"]["model"]["initialization"] == "ImageNet":
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(weights="imagenet", include_top=False)
        total_layers = len(base_model.layers)
        for i in range(5, total_layers):
            if len(base_model.layers[i].weights) > 0:
                print("Setting pre-trained weights: {:.2f}%".format((i / total_layers) * 100), end="\r")
                model.layers[i].set_weights(base_model.layers[i].get_weights())

        # => Replicate filters of first layer as needed
        weights = base_model.layers[4].get_weights()
        available_channels = weights[0].shape[2]
        target_shape = model.layers[4].weights[0].shape
        new_weights = np.zeros(target_shape)

        for i in range(new_weights.shape[2]):
            j = i % available_channels
            new_weights[:, :, i, :] = weights[0][:, :, j, :]

        weights_array = [new_weights]
        if len(weights) > 1:
            weights_array += weights[1:]

        model.layers[4].set_weights(weights_array)
        print("Network initialized with pretrained ImageNet weights")

    elif epoch > 1:
        output_file = config["paths"]["checkpoints"] + "/checkpoint_{epoch:04d}.hdf5"
        model.load_weights(output_file)

    print(model.summary())
    if experiment:
        with experiment.train():
            model.fit(dataset,
                epochs=config["train"]["model"]["epochs"],
                callbacks=callbacks,
                verbose=1,
                validation_data=validation_dataset,
                validation_freq=config["train"]["validation"]["frequency"]
            )
    else:
        model.fit(dataset,
            epochs=config["train"]["model"]["epochs"],
            callbacks=callbacks,
            verbose=1,
            validation_data=validation_dataset,
            validation_freq=config["train"]["validation"]["frequency"]
        )