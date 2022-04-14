from group_9.dlvc.batches import BatchGenerator
from group_9.dlvc.dataset import Subset
from group_9.dlvc.datasets.pets import PetsDataset

import os
import unittest

import cv2
import numpy as np
import numpy.testing
import group_9.dlvc.ops as ops

dataset_path = os.path.join(os.pardir, 'cifar-10-batches-py')

class Part1(unittest.TestCase):
    def test_length_training_set(self):
        d = PetsDataset(dataset_path, Subset.TRAINING)
        self.assertEqual(len(d), 7959)

    def test_length_test_set(self):
        d = PetsDataset(dataset_path, Subset.TEST)
        self.assertEqual(len(d), 2000)

    def test_length_validation_set(self):
        d = PetsDataset(dataset_path, Subset.VALIDATION)
        self.assertEqual(len(d), 2041)

    def test_total_number_of_samples_per_class(self):
        train = PetsDataset(dataset_path, Subset.TRAINING)
        test = PetsDataset(dataset_path, Subset.TEST)
        val = PetsDataset(dataset_path, Subset.VALIDATION)
        labels = []
        for s in train:
            labels.append(s.label)
        for s in test:
            labels.append(s.label)
        for s in val:
            labels.append(s.label)

        self.assertEqual(labels.count(0), 6000, 'There should be 6000 cat samples')
        self.assertEqual(labels.count(1), 6000, 'There should be 6000 dog samples')

    def test_image_shape_and_image_type(self):
        train = PetsDataset(dataset_path, Subset.TRAINING)
        for s in train:
            self.assertTupleEqual(s.data.shape, (32, 32, 3), "Shape of image should be (32, 32, 3)")
            self.assertEqual(s.data.dtype, np.dtype('uint8'))

        test = PetsDataset(dataset_path, Subset.TEST)
        for s in test:
            self.assertTupleEqual(s.data.shape, (32, 32, 3), "Shape of image should be (32, 32, 3)")
            self.assertEqual(s.data.dtype, np.dtype('uint8'))

        val = PetsDataset(dataset_path, Subset.VALIDATION)
        for s in val:
            self.assertTupleEqual(s.data.shape, (32, 32, 3), "Shape of image should be (32, 32, 3)")
            self.assertEqual(s.data.dtype, np.dtype('uint8'))


    def test_first_10_labels_of_training_set(self):
        d = PetsDataset(dataset_path, Subset.TRAINING)
        labels = []
        for i in range(10):
            labels.append(d[i].label)

        self.assertEqual(labels, [0, 0, 0, 0, 1, 0, 0, 0, 0, 1], "First 10 labels should be 0, 0, 0, 0, 1, 0, 0, 0, 0, 1")

    def test_color_channel_order(self):
        train_0002 = PetsDataset(dataset_path, Subset.TRAINING)[2]
        numpy.testing.assert_array_equal(train_0002.data, cv2.imread(os.path.join(os.curdir, 'imgs', 'train_0002.png')))
        self.assertEqual(train_0002.label, 0, "Images are not equal")  # cat

        test_0007 = PetsDataset(dataset_path, Subset.TEST)[7]
        numpy.testing.assert_array_equal(test_0007.data, cv2.imread(os.path.join(os.curdir, 'imgs', 'test_0007.png')))
        self.assertEqual(test_0007.label, 1, "Images are not equal")  # dog

    def test_negative_index_raises_index_error(self):
        d = PetsDataset(dataset_path, Subset.TRAINING)
        with self.assertRaises(IndexError):
            d[-1]

    def test_out_of_bound_index_raises_index_error(self):
        d = PetsDataset(dataset_path, Subset.TRAINING)
        with self.assertRaises(IndexError):
            d[len(d)]

    def test_num_class(self):
        train = PetsDataset(dataset_path, Subset.TRAINING)
        self.assertEqual(train.num_classes(), 2, "num_classes() of training set should return 2")
        test = PetsDataset(dataset_path, Subset.TEST)
        self.assertEqual(test.num_classes(), 2,  "num_classes() of test set should return 2")
        val = PetsDataset(dataset_path, Subset.VALIDATION)
        self.assertEqual(val.num_classes(), 2,  "num_classes() of validation set should return 2")


class Part2(unittest.TestCase):

    def test_ops_type_cast(self):
        o = ops.type_cast(np.float64)
        a = o(np.arange(2, dtype=np.uint8))
        self.assertEqual(a.dtype, np.dtype('float64'))

    def test_ops_add(self):
        n = np.random.randint(-100, 100+1)
        o = ops.add(n)
        a = np.arange(100).reshape((2, 10, -1))
        numpy.testing.assert_array_equal(a + n, o(a))

    def test_ops_mul(self):
        n = np.random.randint(-100, 100+1)
        o = ops.mul(n)
        a = np.arange(100).reshape((2, 10, -1))
        numpy.testing.assert_array_equal(a*n, o(a))

    def test_num_batches_if_batch_size_is_number_of_samples(self):
        d = PetsDataset(dataset_path, Subset.TRAINING)
        bg = BatchGenerator(d, len(d), False)
        self.assertEqual(len(bg), 1)

    def test_num_of_training_batches_if_batch_size_is_500(self):
        d = PetsDataset(dataset_path, Subset.TRAINING)
        bg = BatchGenerator(d, 500, False)
        self.assertEqual(len(bg), 16)

    def test_training_data_and_label_shapes_and_types_if_batch_size_is_500_and_no_ops_applied(self):
        d = PetsDataset(dataset_path, Subset.TRAINING)
        bg = BatchGenerator(d, 500, False)
        num_batches = len(bg)
        for i, b in enumerate(bg):
            if i == (num_batches-1):
                self.assertTupleEqual(b.data.shape, (459, 32, 32, 3))
                self.assertTupleEqual(b.label.shape, (459, ))
                self.assertTupleEqual(b.index.shape, (459, ))
            else:
                self.assertTupleEqual(b.data.shape, (500, 32, 32, 3))
                self.assertTupleEqual(b.label.shape, (500, ))
                self.assertTupleEqual(b.index.shape, (500, ))

            self.assertEqual(b.data.dtype, np.dtype('uint8'))
            self.assertEqual(b.label.dtype, np.dtype('int64'))
            self.assertEqual(b.index.dtype, np.dtype('int64'))

    def test_training_data_and_label_shapes_and_types_if_batch_size_is_500_and_ops_applied(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32),
            ops.add(-127.5),
            ops.mul(1 / 127.5),
        ])

        d = PetsDataset(dataset_path, Subset.TRAINING)
        bg = BatchGenerator(d, 500, False, op)
        num_batches = len(bg)
        for i, b in enumerate(bg):
            if i == (num_batches-1):
                self.assertTupleEqual(b.data.shape, (459, 3072))
                self.assertTupleEqual(b.label.shape, (459, ))
                self.assertTupleEqual(b.index.shape, (459, ))
            else:
                self.assertTupleEqual(b.data.shape, (500, 3072))
                self.assertTupleEqual(b.label.shape, (500, ))
                self.assertTupleEqual(b.index.shape, (500, ))

        self.assertEqual(b.data.dtype, np.dtype('float32'))
        self.assertEqual(b.label.dtype, np.dtype('int64'))
        self.assertEqual(b.index.dtype, np.dtype('int64'))

    def test_batch_generator_invalid_argument_types(self):
        d = PetsDataset(dataset_path, Subset.TRAINING)

        with self.assertRaises(TypeError):
            BatchGenerator(None, len(d), False)

        with self.assertRaises(TypeError):
            BatchGenerator(1, 10, True)

        with self.assertRaises(TypeError):
            BatchGenerator(d, None, False)

        with self.assertRaises(TypeError):
            BatchGenerator(d, 1., False)

        with self.assertRaises(TypeError):
            BatchGenerator(d, len(d), None)

        with self.assertRaises(TypeError):
            BatchGenerator(d, len(d), 0)

        with self.assertRaises(ValueError):
            BatchGenerator(d, len(d)+1, False)


if __name__ == '__main__':
    unittest.main()
