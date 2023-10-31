from _import import *


def loadMatFile(filepath) -> dict:
    data = IO.loadmat(filepath)
    data_dict = {}
    for key, value in data.items():
        if key not in ('__version__', '__globals__', '__header__'):
            value = np.ascontiguousarray(value)
            data_dict[key] = value.astype('float64')
    return data_dict


def split_train_valid_test(data, label, randomstate: int) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    train_data, temp_data, train_label, temp_label = train_test_split(data, label, test_size=0.4,
                                                                      random_state=randomstate, stratify=label)

    valid_data, test_data, valid_label, test_label = train_test_split(temp_data, temp_label, test_size=0.5,
                                                                      random_state=randomstate, stratify=temp_label)

    return train_data, train_label, valid_data, valid_label, test_data, test_label


class datastruct():
    def __init__(self, datasetname: str, filename: str):
        self.datasetname = datasetname
        self.filepath = f'dataset/{filename}.mat'
        self.rawdata = loadMatFile(self.filepath)
        self.mean = np.ndarray([])
        self.std = np.ndarray([])
        self.slices = []
        self.rawdata = self.__normalize()

    def __normalize(self) -> dict:
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
        self.slicenum = slicenum
        data, label, labelmap = self.rawdatatonumpy()
        if (len(data.shape) == 2):
            datamin = np.min(data, axis=0)
            datamax = np.max(data, axis=0)
            datamax = datamax + eps
            assert datamin.shape[0] == data.shape[1]
            assert datamax.shape[0] == data.shape[1]

            num = data.shape[1]
            if (self.slices == []):
                self.__generateslice(datamax, datamin, num, slicenum)
            for i in range(data.shape[1]):
                data[:, i] = np.digitize(data[:, i], self.slices[i, :])
            data = data.astype(int)
            return data, label, labelmap
        elif (len(data.shape) == 3):
            datamin = np.array([np.min(data[:, i, :]) for i in range(data.shape[1])])
            datamax = np.array([np.max(data[:, i, :]) for i in range(data.shape[1])])
            datamax = datamax + eps
            assert datamin.shape[0] == data.shape[1]
            assert datamax.shape[0] == data.shape[1]

            num = data.shape[1]
            if (self.slices == []):
                self.__generateslice(datamax, datamin, num, slicenum)
            for i in range(data.shape[1]):
                data[:, i, :] = np.digitize(data[:, i, :], self.slices[i, :])
            data = data.astype(int)
            return data, label, labelmap
        else:
            raise NotImplementedError('len(data.shape)!=2&&len(data.shape)!=3 Not Implement!')

    def __generateslice(self, datamax, datamin, num, slicenum):
        for min, max in zip(datamin, datamax):
            label_slice = np.linspace(min, max, slicenum)
            self.slices.append(label_slice)
        self.slices = np.array(self.slices)
        assert self.slices.shape == (num, slicenum)
