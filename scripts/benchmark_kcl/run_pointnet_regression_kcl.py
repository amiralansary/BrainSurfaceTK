import os.path as osp

PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
import sys

sys.path.append(PATH_TO_ROOT)

import os
import time
import numpy as np
import csv

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from models.pointnet.src.models.pointnet2_regression_v2 import Net
from models.pointnet.main.pointnet2 import train, test_regression

from scripts.benchmark_kcl.utils import data

PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..') + '/'
PATH_TO_POINTNET = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', '..', 'models', 'pointnet') + '/'

if __name__ == '__main__':

    num_workers = 8
    local_features = ['Curvature', 'MyelinMap', 'Sulc', 'corrThickness']
    global_features = []

    #################################################
    ########### EXPERIMENT DESCRIPTION ##############
    #################################################
    recording = True
    REPROCESS = False

    # data_nativeness = 'native'
    data_compression = "40k"
    data_type = 'white'
    hemisphere = 'right'

    #################################################
    ############ EXPERIMENT DESCRIPTION #############
    #################################################

    # 1. Model Parameters
    ################################################
    lr = 0.001  # 0.001
    batch_size = 2
    comment = 'kcl_left_wm_with_feat_batch_2_lr_001'
    gamma = 0.9875
    scheduler_step_size = 2
    target_class = 'scan_age'
    task = 'regression'
    numb_epochs = 400
    number_of_points = 40000

    ################################################
    ########## INDICES FOR DATA SPLIT #############
    # with open(PATH_TO_POINTNET + 'src/names.pk', 'rb') as f:
    #     indices = pickle.load(f)
    ###############################################
    indices_dir = "/vol/biomedic3/aa16914/dhcp_brain_emma/data/age_prediction/birth_age/data_splits"
    data_folder = "/vol/biomedic3/aa16914/dhcp_brain_emma/data/age_prediction/data/surf_feat_vtp"
    files_ending = "_left_white.sym.40k_fs_LR.surf.shape.label.vtp"

    train_dataset, test_dataset, validation_dataset, train_loader, test_loader, val_loader, num_labels = data(
        data_folder,
        files_ending,
        data_type,
        REPROCESS,
        local_features,
        global_features,
        indices_dir,
        batch_size,
        hemisphere,
        num_workers=8
    )

    print(train_dataset[0])

    if len(local_features) > 0:
        numb_local_features = train_dataset[0].x.size(1)
    else:
        numb_local_features = 0
    numb_global_features = len(global_features)

    # 7. Create the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(numb_local_features, numb_global_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)

    print(f'number of param: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    #################################################
    ############# EXPERIMENT LOGGING ################
    #################################################
    writer = None
    results_folder = None
    if recording:

        # Tensorboard writer.
        writer = SummaryWriter(log_dir='runs/' + task + '/' + comment, comment=comment)

        results_folder = 'runs/' + task + '/' + comment + '/results'
        model_dir = 'runs/' + task + '/' + comment + '/models'

        if not osp.exists(results_folder):
            os.makedirs(results_folder)

        if not osp.exists(model_dir):
            os.makedirs(model_dir)

        with open(results_folder + '/configuration.txt', 'w', newline='') as config_file:
            config_file.write('Learning rate - ' + str(lr) + '\n')
            config_file.write('Batch size - ' + str(batch_size) + '\n')
            config_file.write('Local features - ' + str(local_features) + '\n')
            config_file.write('Global feature - ' + str(global_features) + '\n')
            config_file.write('Number of points - ' + str(number_of_points) + '\n')
            config_file.write('Data res - ' + data_compression + '\n')
            config_file.write('Data type - ' + data_type + '\n')
            # config_file.write('Data nativeness - ' + data_nativeness + '\n')
            # config_file.write('Additional comments - With rotate transforms' + '\n')

        with open(results_folder + '/results.csv', 'w', newline='') as results_file:
            result_writer = csv.writer(results_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow(['Patient ID', 'Session ID', 'Prediction', 'Label', 'Error'])

    #################################################
    #################################################

    best_val_loss = 999

    indices_validation = np.load(os.path.join(indices_dir, 'validation.npy'), allow_pickle=True)[:, 0]

    # MAIN TRAINING LOOP
    for epoch in range(1, numb_epochs + 1):
        start = time.time()
        train(model, train_loader, epoch, device,
              optimizer, scheduler, writer)

        val_mse, val_l1 = test_regression(model, val_loader,
                                          indices_validation, device,
                                          recording, results_folder,
                                          epoch=epoch)

        if recording:
            writer.add_scalar('Loss/val_mse', val_mse, epoch)
            writer.add_scalar('Loss/val_l1', val_l1, epoch)

            print('Epoch: {:03d}, Test loss l1: {:.4f}'.format(epoch, val_l1))
            end = time.time()
            print('Time: ' + str(end - start))
            if val_l1 < best_val_loss:
                best_val_loss = val_l1
                torch.save(model.state_dict(), model_dir + '/model_best.pt')
                print('Saving Model'.center(60, '-'))
            writer.add_scalar('Time/epoch', end - start, epoch)

    indices_test = np.load(os.path.join(indices_dir, 'test.npy'), allow_pickle=True)[:, 0]
    test_regression(model, test_loader, indices_test, device, recording, results_folder, val=False)

    if recording:
        # save the last model
        torch.save(model.state_dict(), model_dir + '/model_last.pt')

        # Eval best model on test
        model.load_state_dict(torch.load(model_dir + '/model_best.pt'))

        with open(results_folder + '/results.csv', 'a', newline='') as results_file:
            result_writer = csv.writer(results_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow(['Best model!'])

        test_regression(model, test_loader, indices_test, device, recording, results_folder, val=False)
