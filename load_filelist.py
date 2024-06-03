import torch.utils.data as data

import os
import os.path
import torch
import numpy as np

class FileListDataLoader(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=None, loader=None):
        self.root      = root
        self.transform = transform
        self.loader    = loader
        self.dataFileList = []
        self.labelList = []

        with open(fileList, 'r') as file:
            for line in file.readlines():
                line.strip('\n')
                line.rstrip()
                information = line.split()
                self.dataFileList.append([float(l) for l in information[1:11]])
                self.labelList.append([float(l) for l in information[11:len(information)]])

    def __getitem__(self, index):
        feature = self.dataFileList[index]
        label = self.labelList[index]

        # x = np.arange(0,6144)
        # plt.plot(x, numpy_array, color='r', label='max', linewidth=1, linestyle='-')
        #
        # plt.show()
        # print(np.max(numpy_array))
        # print(np.min(numpy_array))
        #numpy_array[numpy_array <= 0] = 1

        #numpy_array = np.log10(numpy_array)#/65535
        # print(np.max(numpy_array))
        # print(np.min(numpy_array))
        #numpy_array = (numpy_array - np.min(numpy_array)) / (np.max(numpy_array) - np.min(numpy_array))  # 最大值归一化
        # down_x, down_y = lttb.lttb(range(23399), numpy_array, 4000)
        #dataTensor = torch.FloatTensor(down_y)
        dataTensor = torch.FloatTensor(feature)
        labelTensor = torch.FloatTensor(label)
        return dataTensor, labelTensor

    def __len__(self):
        return len(self.dataFileList)