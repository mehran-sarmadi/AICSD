import os

class Path(object):

    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '../pascal-voc-2012/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/kaggle/input/cityscapes/Cityspaces'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError