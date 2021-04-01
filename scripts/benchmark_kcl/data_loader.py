import os
import os.path as osp

import numpy as np
import pandas as pd
import pyvista as pv
import torch

from torch_geometric.data import Data, DataLoader, InMemoryDataset
import torch_geometric.transforms as T

# from models.pointnet.src.read_meta import read_meta
from tqdm import tqdm


class kclDataset(InMemoryDataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None, data_folder=None,
                 add_features=True, local_features=[], global_feature=[], files_ending=None, reprocess=False,
                 val=False, indices=None, add_faces=False):
        """
        Creates a Pytorch dataset from the .vtk/.vtp brain data.

        :param root: Root path, where processed data objects will be placed
        :param train: If true, save data as training
        :param val: If true and train is false, saves data as validation, test if both are false.
        :param transform: Transformation applied
        :param pre_transform: Pre-transformation applied
        :param pre_filter: Pre-filter applied
        :param data_folder: Path to the data folder with the dataset
        :param add_features: If true, adds all features from .vtp/.vtk files to x in Dataset
        :param reprocess: Flag to reprocess the data even if it was processed before and saved in the root folder.
        :param local_features: Local features that should be added to every point.
        :param global_feature: Global features that should be added to the label for later use.
        :param add_faces: Should the faces be included? Default False, because used with PointNet
               IF THIS IS NOT ZERO, THE PROCESSING IS DONE FOR VALIDATION SET.
        """

        # Train, test, validation
        self.indices_ = indices
        self.train = train
        self.val = val

        # Classes dict. Populated later. Saved in case you need to look this up.
        self.classes = dict()

        # Mapping between features and array number in the files.
        # Old labels: 'drawem', 'corr_thickness', 'myelin_map', 'curvature','sulc'
        self.feature_arrays = {'segmentation': 'segmentation',
                               'corrThickness': 'corrThickness',
                               'MyelinMap': 'MyelinMap',
                               'Curvature': 'Curvature',
                               'Sulc': 'Sulc'}

        # Additional global features
        self.local_features = local_features
        self.global_feature = global_feature

        # Other useful variables
        self.add_features = add_features
        self.add_faces = add_faces

        self.unique_labels = []
        self.num_labels = None

        self.reprocess = reprocess

        # Initialise path to data
        self.data_folder = data_folder

        if files_ending is None:
            self.files_ending = "_right_white.sym.40k_fs_LR.surf.shape.vtp"
        else:
            self.files_ending = files_ending

        super(kclDataset, self).__init__(root, transform, pre_transform, pre_filter)

        # Standard paths to processed data objects (train or test or val)

        if self.train:
            path = self.processed_paths[0]
        elif self.val:
            path = self.processed_paths[2]
        else:
            path = self.processed_paths[1]

        # If processed_paths exist, return without having to process again
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        """A list of files in the raw_dir which needs to be found in order to skip the download."""
        return []

    @property
    def processed_file_names(self):
        """A list of files in the processed_dir which needs to be found in order to skip the processing.
        if self.reprocess, doesn't skip processing"""

        if self.reprocess:
            return ['training.pt', 'test.pt', 'validation.pt', 'a']

        return ['training.pt', 'test.pt', 'validation.pt']

    def download(self):
        """No need to download data."""
        pass

    def process(self):
        """Processes raw data and saves it into the processed_dir."""
        # Read data into huge `Data` list.

        if self.train:
            torch.save(self.process_set(), self.processed_paths[0])
        else:
            if self.val:
                torch.save(self.process_set(), self.processed_paths[2])
            else:
                torch.save(self.process_set(), self.processed_paths[1])

    def get_file_path(self, subject_id):
        file_name = subject_id + self.files_ending
        file_path = os.path.join(self.data_folder, file_name)
        return file_path

    def get_features(self, subject_id, list_features, mesh):
        """
        Returns tensor of features to add in every point.
        :param list_features: list of features to add. Mapping is in self.feature_arrays
        :param mesh: pyvista mesh from which to get the arrays.
        :returns: tensor of features or None if list is empty.
        """
        if list_features:
            # print(self.feature_arrays)
            # print(mesh)
            # print('subject_id', subject_id)
            # print(mesh.point_arrays)
            # change R to L for each hemishpere
            pre_key = subject_id[4:15] + subject_id[19:] + '_R_'
            features = [mesh.get_array(pre_key + key) for key in list_features]
            return torch.tensor(features).t()
        else:
            return None

    def normalise_labels(self, y_tensor, label_mapping):
        """
        Normalises labels in the format necessary for segmentation
        :return: tensor vector of normalised labels ([0, 3, 1, 2, 4, ...])
        """
        # Having received y_tensor, use label_mapping
        temporary_list = []
        for y in y_tensor:
            temporary_list.append(label_mapping[y.item()])
        return torch.tensor(temporary_list)

    def process_set(self):

        """Reads and processes the data. Collates the processed data which is later saved."""
        # 0. Get meta data
        # meta_data = self.labels
        # 1. Initialise the variables
        data_list = []

        # 3. These lists will collect all the information for each patient in order
        lens = []
        xs = []
        poss = []
        ys = []
        faces_list = []

        # 3. Iterate through all patient ids
        # for idx, patient_id in enumerate(meta_data[:, 0]):
        print('Processing patient data for the split...')
        for subject_id, label in tqdm(self.indices_):

            # print('subject', subject_id)
            # print('label', label)
            # print(10 * '=')
            # Get file path to .vtk/.vtp for one patient
            file_path = self.get_file_path(subject_id)

            # if subject_id == 'sub-CC00064XX07_ses-18303':
            #     continue

            # If file exists
            if os.path.isfile(file_path):
                mesh = pv.read(file_path)
                # Get points
                points = torch.tensor(mesh.points)
                if self.add_faces:
                    # Get faces
                    n_faces = mesh.n_cells
                    faces = mesh.faces.reshape((n_faces, -1))
                    faces = torch.tensor(faces[:, 1:].transpose())

                # Features
                x = self.get_features(subject_id, self.local_features, mesh)

                # Generating label based on the task. By default regression.
                y = torch.tensor([[label], ], dtype=torch.float)
                # y = torch.tensor([[float(meta_data[idx, self.meta_column_idx])] + global_x]) #TODO

                # Add the data to the lists
                xs.append(x)
                poss.append(points)
                ys.append(y)

                if self.add_faces:
                    faces_list.append(faces)

            else:
                print('file does not exist:', file_path)

        if self.add_faces:
            # Now add all patient data to the list
            for x, points, y, faces in zip(xs, poss, ys, faces_list):
                # Create a data object and add to data_list
                data = Data(x=x, pos=points, y=y, face=faces)
                data_list.append(data)
        else:
            # Now add all patient data to the list
            for x, points, y in zip(xs, poss, ys):
                # Create a data object and add to data_list
                data = Data(x=x, pos=points, y=y)
                data_list.append(data)

        # Do any pre-processing that is required
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(list(data_list))


if __name__ == '__main__':
    # Path to where the data will be saved.
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/test')

    data_folder = "/vol/biomedic3/aa16914/dhcp_brain_emma/data/age_prediction/data/surf_feat_vtp"
    files_ending = "_right_white.sym.40k_fs_LR.surf.shape.label.vtp"
    # Transformations, scaling and sampling 102 points (doesn't sample faces).
    pre_transform, transform = None, None  # T.NormalizeScale(), T.SamplePoints(1024) #T .FixedPoints(1024)

    # with open('../src/indices_50.pk', 'rb') as f:
    #     indices = pickle.load(f)

    # indices = {'Train': ['sub-CC00576XX16_ses-178200', 'sub-CC00569XX17_ses-170600']}
    TRAIN_DIR = '/vol/biomedic3/aa16914/dhcp_brain_emma/data/age_prediction/birth_age/data_splits/train.npy'
    indices = np.load(TRAIN_DIR, allow_pickle=True)
    # print(indices)
    myDataset = kclDataset(path, train=False, transform=transform, pre_transform=pre_transform, indices=indices,
                           reprocess=False, local_features=['Curvature', 'MyelinMap', 'Sulc', 'corrThickness'],
                           data_folder=data_folder, files_ending=files_ending, val=True)

    # # print(myDataset)
    # # print(myDataset2)
    # print(myDataset[0].x.size(1))
    # print(myDataset[0].y.size(1))

    # # train_loader = DataLoader(myDataset, batch_size=1, shuffle=False)
    # # train_loader2 = DataLoader(myDataset2, batch_size=1, shuffle=False)
    train_loader3 = DataLoader(myDataset, batch_size=1, shuffle=False)
    #
    #  # Printing dataset without sampling points. Will include faces.
    for i, (batch, face, pos, x, y) in enumerate(train_loader3):
        print(i)
        print(batch)
        print(face)
        # print(pos[1].size())
        print(x)
        print(y)
    #     print('_____________')
