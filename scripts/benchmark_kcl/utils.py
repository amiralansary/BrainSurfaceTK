import os
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import numpy as np
from scripts.benchmark_kcl.data_loader import kclDataset

PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..') + '/'


def data(data_folder, files_ending, data_type, REPROCESS, local_features, global_features, indices_dir,
         batch_size, hemisphere, comment='default', num_workers=2):
    """
    Get data loaders and data sets

    :param comment:
    :param hemisphere:
    :param data_folder:
    :param files_ending:
    :param data_type:
    :param REPROCESS:
    :param local_features:
    :param global_features:
    :param indices_dir:
    :param batch_size:
    :param num_workers:
    :return:
    """

    path = osp.join(
        osp.dirname(osp.realpath(__file__)), '..',
        'data/' + 'scan_age' + f'/{comment}/{hemisphere}/{data_type}')

    print('root path', path)

    indices_train = np.load(os.path.join(indices_dir, 'train.npy'), allow_pickle=True)
    indices_test = np.load(os.path.join(indices_dir, 'test.npy'), allow_pickle=True)
    indices_validation = np.load(os.path.join(indices_dir, 'validation.npy'), allow_pickle=True)

    # Transformations
    transform = T.Compose([
        # T.RandomTranslate(0.1),
        # T.RandomFlip(0, p=0.3),
        # T.RandomFlip(1, p=0.1),
        # T.RandomFlip(2, p=0.3),
        # T.FixedPoints(500, replace=False), #32492  16247
        T.RandomRotate(360, axis=0),
        T.RandomRotate(360, axis=1),
        T.RandomRotate(360, axis=2)
    ])

    pre_transform = T.NormalizeScale()
    print('Starting dataset processing...')
    train_dataset = kclDataset(path, train=True, transform=transform, pre_transform=pre_transform, reprocess=REPROCESS,
                               local_features=local_features, global_feature=global_features,
                               val=False, indices=indices_train,
                               data_folder=data_folder,
                               files_ending=files_ending)

    test_dataset = kclDataset(path, train=False, transform=transform, pre_transform=pre_transform, reprocess=REPROCESS,
                              local_features=local_features, global_feature=global_features,
                              val=False, indices=indices_test,
                              data_folder=data_folder,
                              files_ending=files_ending)

    validation_dataset = kclDataset(path, train=False, transform=transform, pre_transform=pre_transform, reprocess=REPROCESS,
                                    local_features=local_features, global_feature=global_features,
                                    val=True, indices=indices_validation,
                                    data_folder=data_folder,
                                    files_ending=files_ending)

    num_labels = train_dataset.num_labels

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataset, test_dataset, validation_dataset, train_loader, test_loader, val_loader, num_labels


if __name__ == '__main__':
    pass
    # # Model Parameters
    # lr = 0.001
    # batch_size = 8
    # num_workers = 2
    #
    # local_features = ['corr_thickness', 'myelin_map', 'curvature', 'sulc']
    # global_features = None
    # target_class = 'gender'
    # task = 'segmentation'
    # # number_of_points = 12000
    #
    # test_size = 0.09
    # val_size = 0.1
    # reprocess = False
    #
    # data = "reduced_50"
    # type_data = "inflated"
    #
    # log_descr = "LR=" + str(lr) + '\t\t'\
    #           + "Batch=" + str(batch_size) + '\t\t'\
    #           + "Num Workers=" + str(num_workers) + '\t'\
    #           + "Local features:" + str(local_features) + '\t'\
    #           + "Global features:" + str(global_features) + '\t'\
    #           + "Data used: " + data + '_' + type_data + '\t'\
    #           + "Split class: " + target_class
    #
    # save_to_log(log_descr)
