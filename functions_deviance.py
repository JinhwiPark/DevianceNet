import os
from PIL import Image
from torch.utils import data
import torch
import shutil
from collections import OrderedDict
import torch.nn.functional as F


class DatasetDeviance(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_path, frames, transform=None, partition=False, direction=False):
        "Initialization"
        self.folders = []
        self.labels = []

        if type(data_path) == type(''):
            folders = [os.path.join(data_path, i) for i in os.listdir(data_path)]
            self.labels += [int(f[len(data_path) + 8]) - 1 for f in folders]
            self.folders += folders
        else:
            for d in data_path:
                l = len(d)
                folders = [os.path.join(d, i) for i in os.listdir(d)]
                self.labels += [int(f[l + 8]) - 1 for f in folders]
                self.folders += folders
        if partition:
            from sklearn.model_selection import train_test_split
            train_list, test_list, train_label, test_label = train_test_split(self.folders, self.labels, shuffle=True,
                                                                              test_size=partition / 100.0)
            self.labels = test_label
            self.folders = test_list
        if direction:
            if len(direction) == 2:
                directions = [direction]
            else:
                directions = direction.split(',')
            new_folders = []
            new_labels = []
            for i_ in directions:
                for i in range(len(self.labels)):
                    if i_ in self.folders[i]:
                        new_folders.append(self.folders[i])
                        new_labels.append(self.labels[i])
            self.folders = new_folders
            self.labels = new_labels
        assert len(self.labels) == len(self.folders)
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, use_transform):
        X = []

        for i in self.frames:  # 0~15
            try:
                image = Image.open(os.path.join(path, 'frame{:06d}.jpg'.format(i)))
            except:
                try:
                    image = Image.open(
                        os.path.join(path, 'frame{:06d}.jpg'.format(1 + i - len(os.listdir(path)))))
                except:
                    image = Image.open(os.path.join(path, 'frame000000.jpg'))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        folder = self.folders[index]
        # Load data
        X = self.read_images(folder, self.transform)
        y = torch.LongTensor([self.labels[index]])

        X = X.permute(1, 0, 2, 3)
        return X, y #, folder


def load_pth(pth_path):
    # original saved file with DataParallel
    state_dict = torch.load(pth_path, map_location=torch.device('cpu'))
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(
        ".", "..", ".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*",
        "*pretrained*", 'outputs', 'save_model', 'superpoint_model')

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))


def H_loss(cl_, de, label, device, args):
    pred_de = torch.ones(de.size(0)).to(device).long()
    pred_de[label != 4] = 0.
    DEloss = F.cross_entropy(de, pred_de)

    h_1 = args.h_1 * args.lambda_2  # 0.909
    h_2 = args.h_2 * args.lambda_2  # 0.088
    h_3 = args.h_3 * args.lambda_2  # 0.003

    CLloss = torch.zeros(1).to(device)
    cl_ = F.softmax(cl_, dim=1)
    for i in range(label.size(0)):
        if label[i] == 0:  # class 1
            CLloss += torch.log(cl_[i][label[i]]) + h_1 * torch.log(cl_[i][label[i] + 1]) + \
                      h_2 * torch.log(cl_[i][label[i] + 2]) + h_3 * torch.log(cl_[i][label[i] + 3])
        elif label[i] == 1:  # class 2
            CLloss += torch.log(cl_[i][label[i]]) + h_1 * (torch.log(cl_[i][label[i] + 1])) + \
                      h_2 * (torch.log(cl_[i][label[i] + 2])) + h_3 * torch.log(cl_[i][label[i] + 3])
        elif label[i] == 2:  # class 3
            CLloss += torch.log(cl_[i][label[i]]) + h_1 * (torch.log(cl_[i][label[i] + 1])) + \
                      h_2 * (torch.log(cl_[i][label[i] + 2]))
        elif label[i] == 3:  # class 4
            CLloss += torch.log(cl_[i][label[i]]) + h_1 * (torch.log(cl_[i][label[i] + 1]))
        elif label[i] == 4:  # class 5
            CLloss += torch.log(cl_[i][label[i]])
    CLloss = -CLloss / label.size(0)
    return args.lambda_1 * DEloss, CLloss
