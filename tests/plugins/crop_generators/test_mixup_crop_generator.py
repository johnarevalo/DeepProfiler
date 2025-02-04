import pytest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import tensorflow as tf
import os
import skimage

import plugins.crop_generators.mixup_crop_generator
import deepprofiler.imaging.cropping
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target

tf.compat.v1.disable_v2_behavior()


@pytest.fixture(scope="function")
def crop_generator(config, dataset):
    return plugins.crop_generators.mixup_crop_generator.GeneratorClass(config, dataset)

@pytest.fixture(scope="function")
def prepared_crop_generator(crop_generator, out_dir):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data["R"][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data["G"][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data["B"][i // 3]), images[:, :, i + 2])
    crop_generator.build_input_graph()
    return crop_generator


@pytest.fixture(scope="function")
def mixup():
    test_alpha = 1
    test_table_size = 500
    test_crop_shape = [(16,16,3),()]
    test_target_size = 500
    return plugins.crop_generators.mixup_crop_generator.Mixup(
                                                     test_table_size,
                                                     test_crop_shape,
                                                     test_target_size,
                                                     test_alpha
                                                     )


def test_init_mixup(mixup):
    test_table_size = 500
    test_target_size = 500
    test_alpha = 1
    assert mixup.table_size == test_table_size
    assert mixup.target_size == test_target_size
    np.testing.assert_array_equal(mixup.crops,np.zeros( (test_table_size, 16, 16, 3) ))
    assert_frame_equal(mixup.labels,pd.DataFrame(data=np.zeros((test_table_size), dtype=np.int32),  columns=["target"]))
    assert mixup.pointer == 0
    assert mixup.ready == False
    assert mixup.alpha == test_alpha


def test_add_crops(mixup):
    test_table_size = 500
    test_crops = []
    test_load = 50
    np.random.seed(50)
    for num in range(test_load):
        test_crops.append(np.random.randint(256, size=(16, 16, 3), dtype=np.uint16))
    test_crops = np.array(test_crops)
    test_labels = np.ones( (test_load,2) )
    test_labels[:,1] = np.array(test_load*[2])
    mixup.add_crops(test_crops, test_labels)
    assert mixup.crops.shape == (test_table_size,16,16,3)
    assert mixup.labels.shape == (test_table_size,1)
    np.testing.assert_array_equal(mixup.crops,np.concatenate((test_crops,np.zeros( (test_table_size-test_load, 16, 16, 3) ))))
    assert_frame_equal(mixup.labels,pd.DataFrame(data=np.concatenate((np.ones((test_load), dtype=np.int64),np.zeros((test_table_size-test_load), dtype=np.int64))),columns=["target"]))
    assert mixup.pointer == test_load
    test_load = 500
    test_crops = []
    np.random.seed(500)
    for num in range(test_load):
        test_crops.append(np.random.randint(256, size=(16, 16, 3), dtype=np.uint16))
    test_crops = np.array(test_crops)
    test_labels = np.ones( (test_load,3) )
    test_labels[:,2] = np.array(test_load*[4])
    mixup.add_crops(test_crops, test_labels)
    assert mixup.pointer == test_load - (mixup.table_size - 50) #works because test_load > cropset.table_size by 50
    assert mixup.ready == True
    np.testing.assert_array_equal(mixup.crops,np.concatenate((test_crops[450:500],test_crops[0:450] )))
    assert_frame_equal(mixup.labels,pd.DataFrame(data=np.array((test_load*[2]), dtype=np.int64),columns=["target"]))


def test_batch_mixup(mixup):
    test_load = 500
    test_crops = []
    mixup.alpha = 0.3
    test_seed = 600
    np.random.seed(test_seed)
    for num in range(test_load):
        test_crops.append(np.random.randint(256, size=(16, 16, 3), dtype=np.uint16))
    test_crops = np.array(test_crops)
    test_labels = np.ones( (test_load,3) )
    test_labels[:,2] = np.array(test_load*[4])
    mixup.add_crops(test_crops, test_labels)
    test_batch_size = 50
    
    assert mixup.batch(test_batch_size)[0].shape == (test_batch_size, mixup.crops.shape[1], mixup.crops.shape[2], mixup.crops.shape[3])
    assert mixup.batch(test_batch_size)[1].shape == (test_batch_size, mixup.target_size)
    np.testing.assert_array_equal(mixup.labels["target"].unique(),np.array(([2]), dtype=np.int64))
    
    expected_data = np.zeros( (test_batch_size, mixup.crops.shape[1], mixup.crops.shape[2], mixup.crops.shape[3]) ) 
    np.random.seed(test_seed)
    for i in range(test_batch_size):
        test_lam = np.random.beta(mixup.alpha, mixup.alpha)
        test_sample = mixup.labels.sample(n=2,random_state=test_seed)
        test_idx = test_sample.index.tolist()
        expected_data[i,:,:,:] = test_lam*mixup.crops[test_idx[0],...] + (1. - test_lam)*mixup.crops[test_idx[1],...]
    np.testing.assert_array_equal(mixup.batch(test_batch_size,seed=test_seed)[0],expected_data)
    
    expected_labels = np.zeros((test_batch_size, mixup.target_size))
    expected_labels[:,2] = 1.0
    np.testing.assert_array_equal(mixup.batch(test_batch_size,seed=test_seed)[1],expected_labels)


def test_mixup_crop_generator():
    assert issubclass(plugins.crop_generators.mixup_crop_generator.GeneratorClass, deepprofiler.imaging.cropping.CropGenerator)
    assert issubclass(plugins.crop_generators.mixup_crop_generator.SingleImageGeneratorClass, deepprofiler.imaging.cropping.SingleImageCropGenerator)


def test_start(prepared_crop_generator):  # includes test for training queues
    sess = tf.compat.v1.Session()
    prepared_crop_generator.start(sess)
    assert not prepared_crop_generator.coord.joined
    assert not prepared_crop_generator.exception_occurred
    assert len(prepared_crop_generator.queue_threads) == prepared_crop_generator.config["train"]["sampling"]["workers"]
    assert prepared_crop_generator.batch_size == prepared_crop_generator.config["train"]["model"]["params"]["batch_size"]
    # TODO check this number next line, used to be 3, test is passed atm
    assert prepared_crop_generator.target_sizes[0] == 4
    assert isinstance(prepared_crop_generator.mixer, plugins.crop_generators.mixup_crop_generator.Mixup)
    prepared_crop_generator.stop(sess)


def test_generate(prepared_crop_generator):
    sess = tf.compat.v1.Session()
    prepared_crop_generator.start(sess)
    generator = prepared_crop_generator.generate(sess)
    prepared_crop_generator.ready_to_sample = True
    test_steps = 3
    for i in range(test_steps):
        data = next(generator)
        assert np.array(data[0]).shape == (prepared_crop_generator.config["train"]["model"]["params"]["batch_size"],
                                           prepared_crop_generator.config["dataset"]["locations"]["box_size"],
                                           prepared_crop_generator.config["dataset"]["locations"]["box_size"],
                                           len(prepared_crop_generator.config["dataset"]["images"]["channels"]))
        assert data[1].shape == (prepared_crop_generator.config["train"]["model"]["params"]["batch_size"], prepared_crop_generator.dset.targets[0].shape[1])
    prepared_crop_generator.stop(sess)
