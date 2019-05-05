"""Test cases for the DataSet class."""
import os
import shutil
from unittest import TestCase
from .._dataset import DataSet


# extract the path to this directory
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '{}')


class __init__ShouldRaiseErrorOnMissingPositionalArgumented(TestCase):
    def test(self):
        self.assertRaises(TypeError, DataSet)


class __init__ShouldRaiseErrorOnInvalidTypeForArg_path_(TestCase):
    def test(self):
        self.assertRaises(TypeError, DataSet, 0)


class __init__ShouldRaiseErrorOnInvalidValueForArg_path_(TestCase):
    def test(self):
        self.assertRaises(ValueError, DataSet, 'not a valid path')


class __init__ShouldRaiseErrorOnMissingXDirectoryIn_path_(TestCase):
    def test(self):
        self.assertRaises(ValueError, DataSet, PATH.format('missing_X'))


class __init__ShouldRaiseErrorOnMissingYDirectoryIn_path_(TestCase):
    def test(self):
        self.assertRaises(ValueError, DataSet, PATH.format('missing_y'))


class __init__ShouldRaiseErrorOnMissingLabelColorsFileIn_path_(TestCase):
    def test(self):
        self.assertRaises(ValueError, DataSet, PATH.format('missing_label_colors'))


class __init__ShouldCreateDataSetWithNoMapping(TestCase):
    def test(self):
        path_ = PATH.format('valid')
        y_full = os.path.join(path_, 'y_full')
        try:
            shutil.rmtree(y_full)
        except FileNotFoundError:
            pass
        try:
            DataSet(path_)
        except Exception:
            self.fail('initializer raised unexpected error')
        self.assertTrue(os.path.isdir(y_full))


class __init__ShouldCreateDataSetWithMapping(TestCase):
    def test(self):
        path_ = PATH.format('valid')
        y_8 = os.path.join(path_, 'y_ff6d2b7169d04fac6d51496fea893432')
        try:
            shutil.rmtree(y_8)
        except FileNotFoundError:
            pass
        mapping = os.path.join(path_, '8_class.txt')
        try:
            DataSet(path_, mapping=mapping)
        except Exception:
            self.fail('/initializer raised unexpected error')
        self.assertTrue(os.path.isdir(y_8))


def valid_dataset(ignored_labels=[], data_format='channels_last') -> DataSet:
    path_ = PATH.format('valid')
    dataset = DataSet(path_,
        mapping=os.path.join(path_, '8_class.txt'),
        ignored_labels=ignored_labels,
        data_format=data_format,
    )

    return dataset


class discrete_to_rgb_mapShouldReturn(TestCase):
    def test(self):
        dataset = valid_dataset()
        self.assertTrue(0 in dataset.discrete_to_rgb_map)
        self.assertEqual((128, 0, 0), dataset.discrete_to_rgb_map[0])
        self.assertTrue(7 in dataset.discrete_to_rgb_map)
        self.assertEqual((0, 0, 0), dataset.discrete_to_rgb_map[7])


class discrete_to_label_mapShouldReturn(TestCase):
    def test(self):
        dataset = valid_dataset()
        self.assertTrue(0 in dataset.discrete_to_label_map)
        self.assertEqual('Building', dataset.discrete_to_label_map[0])
        self.assertTrue(7 in dataset.discrete_to_label_map)
        self.assertEqual('Void', dataset.discrete_to_label_map[7])


class label_to_discrete_mapShouldReturn(TestCase):
    def test(self):
        dataset = valid_dataset()
        self.assertTrue('Building' in dataset.label_to_discrete_map)
        self.assertEqual(0, dataset.label_to_discrete_map['Building'])
        self.assertTrue('Void' in dataset.label_to_discrete_map)
        self.assertEqual(7, dataset.label_to_discrete_map['Void'])


class ignored_codesShouldReturn(TestCase):
    def test(self):
        dataset = valid_dataset()
        self.assertEqual([], dataset.ignored_codes)
        dataset = valid_dataset(ignored_labels=['Void'])
        self.assertEqual([7], dataset.ignored_codes)


class metadataShouldReturn(TestCase):
    def test(self):
        dataset = valid_dataset()
        self.assertTrue('label' in dataset.metadata.columns)
        self.assertTrue('rgb' in dataset.metadata.columns)
        self.assertTrue('label_used' in dataset.metadata.columns)
        self.assertTrue('rgb_draw' in dataset.metadata.columns)
        self.assertTrue('code' in dataset.metadata.columns)


class shapeShouldReturn(TestCase):
    def test(self):
        dataset = valid_dataset()
        self.assertEqual((224, 224, 3), dataset.shape)


class num_classesShouldReturn(TestCase):
    def test(self):
        dataset = valid_dataset()
        self.assertEqual(8, dataset.num_classes)


class class_weightsShouldReturn(TestCase):
    def test(self):
        dataset = valid_dataset()
        self.assertEqual([
            0.2261608129740443,
            0.5175716672708787,
            11.160279769601503,
            0.8224870268818603,
            1.2752249832102083,
            0.6507565501996334,
            5.845729942737517,
            1.299629031374911
        ], list(dataset.class_weights))


class class_maskShouldReturn(TestCase):
    def test(self):
        dataset = valid_dataset()
        self.assertEqual([1, 1, 1, 1, 1, 1, 1, 1], list(dataset.class_mask))
        dataset = valid_dataset(ignored_labels=['Void'])
        self.assertEqual([1, 1, 1, 1, 1, 1, 1, 0], list(dataset.class_mask))


class samplesShouldReturn(TestCase):
    def test(self):
        dataset = valid_dataset()
        self.assertEqual({'train': 3, 'val': 2}, dataset.samples)


class steps_per_epochShouldReturn(TestCase):
    def test(self):
        dataset = valid_dataset()
        self.assertEqual(1, dataset.steps_per_epoch)


class validation_stepsShouldReturn(TestCase):
    def test(self):
        dataset = valid_dataset()
        self.assertEqual(2, dataset.validation_steps)


class test_stepsShouldReturn(TestCase):
    def test(self):
        dataset = valid_dataset()
        self.assertRaises(ValueError, lambda: dataset.test_steps)


class test_ShouldCreateGenerators(TestCase):
    def test(self):
        dataset = valid_dataset()
        generators = dataset.generators()
        self.assertIn('train', generators)
        self.assertIn('val', generators)
        self.assertNotIn('test', generators)


class test_ShouldGenerateBatch(TestCase):
    def test(self):
        dataset = valid_dataset()
        generators = dataset.generators()
        x, y = next(generators['train'])
        self.assertEqual((3, 224, 224, 3), x.shape)
        self.assertEqual((3, 224, 224, 8), y.shape)
        x, y = next(generators['val'])
        self.assertEqual((1, 224, 224, 3), x.shape)
        self.assertEqual((1, 224, 224, 8), y.shape)


class test_ShouldGenerateBatchChannelsFirst(TestCase):
    def test(self):
        dataset = valid_dataset(data_format='channels_first')
        generators = dataset.generators()
        x, y = next(generators['train'])
        self.assertEqual((3, 3, 224, 224), x.shape)
        self.assertEqual((3, 8, 224, 224), y.shape)
        x, y = next(generators['val'])
        self.assertEqual((1, 3, 224, 224), x.shape)
        self.assertEqual((1, 8, 224, 224), y.shape)
