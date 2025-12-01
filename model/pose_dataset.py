from abc import ABCMeta

import torch

from sensfloor.model.utils import SIGNAL_Z

joint_count = 21
roi_x_length = 12
roi_y_length = 12

def transform_joints_to_labels(joints: torch.Tensor) -> torch.Tensor:
    """
    joints as tenser[joint_count](x,y,z)
    """
    labels = torch.zeros((joint_count, roi_x_length, roi_y_length, SIGNAL_Z))
    for joint, i in enumerate(joints):
        x, y, z = joint
        labels[i][x][y][z] = 1

    return labels

class PoseDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.input = torch.randn(size=(1024, 64, roi_x_length, roi_y_length)) #TODO actual data
        self.labels = torch.zeros(size=(1024, joint_count, 3)) #TODO actual label
        self.labels[:,:,0] = 2
        self.labels[:,:,1] = 2
        self.labels[:,:,2] = 9

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index: int):
        return self.input[index], self.labels[index]