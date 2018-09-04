import os
import numpy as np

def get_path_list(rootdir):
    '''
    :param rootdir: path 图片路径
    :return: file list
    '''
    FilePathList = []
    for fpathe, dirs, fs in os.walk(rootdir):
        for f in fs:
            FilePath = os.path.join(fpathe, f)
            if os.path.isfile(FilePath):
                FilePathList.append(FilePath)
    return FilePathList

def get_npy_data_by_batch(FilePathList,batch_size):
    '''
    :param FilePathList: FILElIST
    :param batch_size:   小批数量
    :return: (batch_size,7,7,12) (batch_size,1)
    '''
    # FilePathList = get_path_list(x_path)
    FilePathList_Len = len(FilePathList)
    # print(FilePathList_Len)
    FileIndexs = np.random.randint(0, FilePathList_Len, batch_size)
    # print(FileIndexs)

    i = 0
    for file_i in FileIndexs:
        npz = np.load(FilePathList[file_i])
        features = npz['features']
        label = npz['label']
        if i == 0:
            features_np_list = features
            lable_np_list = label
        else:
            features_np_list = np.vstack((features_np_list,features))
            lable_np_list = np.vstack((lable_np_list,label))
        i = i + 1
    return features_np_list,lable_np_list




def generate_batch_data_random(x_path,batch_size):
    """逐步提取batch数据到显存，降低对显存的占用"""

    FilePathList = get_path_list(x_path)
    while (True):
        features_np_list, lable_np_list = get_npy_data_by_batch(FilePathList, batch_size)
        (train_x,train_y) = features_np_list, lable_np_list
        yield (train_x,train_y)


if __name__ == '__main__':
    i = 1
    for data in generate_batch_data_random('data_np',2):
        print(type(data))
        i = i + 1
        if i>5:
            break