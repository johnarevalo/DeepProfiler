import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

tf.compat.v1.disable_v2_behavior()


#################################################
# CROPPING AND TRANSFORMATION OPERATIONS
#################################################

def random_crop(image):
    w,h,c = image.shape
    
    size = tf.random.uniform([1], minval=int(w.value*0.6), maxval=w, dtype=tf.int32)
    crop = tf.image.random_crop(image, [size[0],size[0],c])
    
    result = tf.image.resize(
        tf.expand_dims(crop, 0), [w,h], method="bicubic"
    )
    
    return result[0,...]


def random_illumination(image):
    # Make channels independent images
    numchn = image.shape[-1].value
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


def random_flips(image):
    # Horizontal flips
    augmented = tf.image.random_flip_left_right(image)

    # 90 degree rotations
    angle = tf.random.uniform([1], minval=0, maxval=4, dtype=tf.int32)
    augmented = tf.image.rot90(augmented, angle[0])
    
    return augmented

def random_rotate(image):
    w, h, c = image.shape
    image = tfa.image.rotate(image, np.pi / tf.random.uniform(shape=[], minval=1, maxval=10, dtype=tf.float32))
    image = tf.image.central_crop(image, 0.7)
    return tf.image.resize(image, (w, h))


def augment(image):
    if tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(0.5, tf.float32)):
        augm = random_crop(image)
    else:
        augm = random_rotate(image)

    augm = random_illumination(augm)
    augm = random_flips(augm)

    return augm


def old_augment(crop):
    with tf.compat.v1.variable_scope("augmentation"):
        # Horizontal flips
        augmented = tf.image.random_flip_left_right(crop)

        # 90 degree rotations
        angle = tf.compat.v1.random_uniform([1], minval=0, maxval=4, dtype=tf.int32)
        augmented = tf.image.rot90(augmented, angle[0])

        # 5 degree inclinations
        angle = tf.compat.v1.random_normal([1], mean=0.0, stddev=0.03 * np.pi, dtype=tf.float32)
        augmented = tensorflow_addons.image.rotate(augmented, angle[0], interpolation="BILINEAR")

        # Translations (3% movement in x and y)
        offsets = tf.compat.v1.random_normal([2],
                                             mean=0,
                                             stddev=int(crop.shape[0].value * 0.03)
                                             )
        augmented = tensorflow_addons.image.translate(augmented, translations=offsets)

        # Illumination changes (10% changes in intensity)
        illum_s = tf.compat.v1.random_normal([1], mean=1.0, stddev=0.1, dtype=tf.float32)
        illum_t = tf.compat.v1.random_normal([1], mean=0.0, stddev=0.1, dtype=tf.float32)
        augmented = augmented * illum_s + illum_t

    return augmented


def augment_multiple(crops, parallel=None):
    print("+")
    return tf.map_fn(augment, crops, parallel_iterations=parallel, dtype=tf.float32)


## A layer for GPU accelerated augmentations

#AugmentationLayer = tf.keras.layers.Lambda(augment_multiple)

class AugmentationLayer(tf.compat.v1.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.is_training = True
        super(AugmentationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        return

    def call(self, input_tensor):
        if self.is_training:
            return augment_multiple(input_tensor)
        else:
            return input_tensor
