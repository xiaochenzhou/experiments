import numpy as np
from sklearn import svm, datasets
import sklearn
from Data_load import read_UCI_data, read_2D, data_clean

def data_more(element, array):
    true_table = (array == element)

    size = np.size(element)

    sum_table = np.sum(true_table, axis=1)

    k = np.sum(sum_table==size)
    index = np.argwhere((sum_table==size))
    if (k>1):
        return True
    else:
        return False

# Training_set = sklearn.datasets.load_svmlight_file('skin_nonskin.txt')
# Training_set = sklearn.datasets.load_svmlight_file('ijcnn1.tr')
# Training_set = sklearn.datasets.load_svmlight_file('phishing.txt')
Training_set = sklearn.datasets.load_svmlight_file('covtype.libsvm.binary.scale')
# Training_set = sklearn.datasets.load_svmlight_file('a8a.txt')

Train_data = np.array(Training_set[0].todense())
Train_label = Training_set[1]

# save_data = Train_data
# save_label = Train_label
save_data, save_label = data_clean(Train_data, Train_label)

# np.save("skin_nonskin_data.npy",save_data)
# np.save("skin_nonskin_label.npy",save_label)

# np.save("a8a_data.npy",save_data)
# np.save("a8a_label.npy",save_label)

np.save("covtype_data.npy",save_data)
np.save("covtype_label.npy",save_label)
