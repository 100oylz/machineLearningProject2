import numpy as np

from _import import *

def loadMatFile(filepath: str) -> dict:
    """
    从MATLAB文件中加载数据。

    参数：
        filepath (str)：MATLAB文件的路径。

    返回：
        dict：包含加载数据的字典。
    """
    data = IO.loadmat(filepath)
    data_dict = {}
    for key, value in data.items():
        if key not in ('__version__', '__globals__', '__header__'):
            value = np.ascontiguousarray(value)
            data_dict[key] = value.astype('float64')
    return data_dict

class datastruct():
    def __init__(self, datasetname: str, filename: str):
        """
        初始化datastruct对象。

        参数：
            datasetname (str)：数据集的名称。
            filename (str)：包含数据集的MATLAB文件的文件名。
        """
        self.datasetname = datasetname
        self.filepath = f'dataset/{filename}.mat'
        self.rawdata = loadMatFile(self.filepath)
        self.mean = np.ndarray([])
        self.std = np.ndarray([])
        self.slices = []
        self.rawdata = self.__normalize()

    def __normalize(self) -> dict:
        """
        对原始数据进行归一化处理。

        返回：
            dict：包含归一化后数据的字典。
        """
        newdatadict = {}
        for key, value in self.rawdata.items():
            valuemean = np.mean(value, axis=0)
            valuestd = np.std(value, axis=0)
            if not np.any(valuestd == 0):
                newvalue = (value - valuemean) / valuestd
            else:
                newvalue = value
            self.mean = np.append(self.mean, valuemean)
            self.std = np.append(self.std, valuestd)
            newdatadict[key] = newvalue
        return newdatadict

    def rawdatatonumpy(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        将原始数据转换为NumPy数组。

        返回：
            np.ndarray：数据。
            np.ndarray：标签。
            np.ndarray：标签映射。
        """
        data = []
        label = []
        labelmap = []
        pointer = 0
        for key, value in self.rawdata.items():
            labelmap.append(key)
            for item in value:
                label.append(pointer)
                data.append(item)
            pointer += 1
        return np.array(data), np.array(label), np.array(labelmap)

    def discrete(self, slicenum=100, eps=1e-18) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        对数据进行离散化处理。

        参数：
            slicenum (int, 可选)：离散化的分段数。默认为100。
            eps (float, 可选)：避免除零的小值。默认为1e-18。

        返回：
            np.ndarray：离散化后的数据。
            np.ndarray：标签。
            np.ndarray：标签映射。
        """
        self.slicenum = slicenum
        data, label, labelmap = self.rawdatatonumpy()
        print(data.shape)
        if len(data.shape) == 2:
            datamin = np.min(data, axis=0)  # 每个特征的最小值
            datamax = np.max(data, axis=0)  # 每个特征的最大值
            datamax = datamax + eps
            assert datamin.shape[0] == data.shape[1]
            assert datamax.shape[0] == data.shape[1]

            num = data.shape[1]
            if not self.slices:
                self.__generateslice(datamax, datamin, num, slicenum)
            for i in range(data.shape[1]):
                data[:, i] = np.digitize(data[:, i], self.slices[i, :])
            data = data.astype(int)
            return data, label, labelmap

        elif len(data.shape) == 3:
            number = data.shape[0]
            feature = data.shape[2]

            data_mean = np.zeros((number,feature))
            data_mean = np.array(data_mean)

            for i in range(number):
                item = data[1,:,:]
                mean = np.mean(item,axis = 1)

                for j in range(len(mean)):
                    data_mean[i,j] = mean[j]
            datamin = np.min(data_mean, axis=0)  # 每个特征的最小值
            datamax = np.max(data_mean, axis=0)  # 每个特征的最大值
            datamax = datamax + eps
            assert datamin.shape[0] == data_mean.shape[1]
            assert datamax.shape[0] == data_mean.shape[1]

            num = data_mean.shape[1]
            if not self.slices:
                self.__generateslice(datamax, datamin, num, slicenum)
            for i in range(data_mean.shape[1]):
                data_mean[:, i] = np.digitize(data_mean[:, i], self.slices[i, :])
            data = data_mean.astype(int)
            return data, label, labelmap

        # data = data.transpose((-1, -2))
        # datamin = np.array([np.min(data[:, :, i]) for i in range(data.shape[1])])
        # datamax = np.array([np.max(data[:, :, i]) for i in range(data.shape[1])])
        # datamax = datamax + eps
        # assert datamin.shape[0] == data.shape[2]
        # assert datamax.shape[0] == data.shape[2]
        #
        # num = data.shape[1]
        # if not self.slices:
        #     self.__generateslice(datamax, datamin, num, slicenum)
        # for i in range(data.shape[1]):
        #     data[:, :, i] = np.digitize(data[:, :, i], self.slices[i, :])
        # data: np.ndarray = data.astype(int)
        # return data, label, labelmap

        else:
            raise NotImplementedError('len(data.shape)!=2&&len(data.shape)!=3 Not Implement!')

    def __generateslice(self, datamax: np.ndarray, datamin: np.ndarray, num: int, slicenum: int):
        """
        生成离散化的切片。

        参数：
            datamax (np.ndarray)：每个特征的最大值。
            datamin (np.ndarray)：每个特征的最小值。
            num (int)：特征数量。
            slicenum (int)：离散化的分段数。
        """
        for min, max in zip(datamin, datamax):
            label_slice = np.linspace(min, max, slicenum)  # 在最大值和最小值之间切出100片进行分段
            self.slices.append(label_slice)
        self.slices = np.array(self.slices)
        assert self.slices.shape == (num, slicenum)

if __name__ == '__main__':
    OCD_fMRI = datastruct('OCD_fMRI', 'OCD_90_200_fMRI')
    data, label, labelmap = OCD_fMRI.discrete(slicenum=100)

    # PPMI = datastruct('PPMI', 'PPMI')
    # data, label, labelmap = PPMI.discrete(slicenum=100)
    print(data.shape)

