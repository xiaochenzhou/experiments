import numpy as np
from sklearn import svm, datasets
from Data_dist import data_partition, kmeans_partition, k_means_random_partition
import Edge_node as Ed
import sklearn
import Central_node as Ce
from sklearn.utils import shuffle
import copy

from sklearn import preprocessing
import matplotlib.pyplot as plt
from plot_utility import SVM_plot
from Data_load import read_UCI_data, read_2D, read_skin


def array_compare(element, array):
    true_table = (array == element)

    if (np.size(element) in np.sum(true_table, axis=1)):
        return True
    else:
        return False


def SV_compare(old_SV, new_SV):
    if (np.size(old_SV, axis=0) != np.size(new_SV, axis=0)):
        return False
    for i in range(np.size(new_SV, axis=0)):
        if not array_compare(new_SV[i], old_SV):
            return False

    return True


def SV_diff_count(global_SV, dis_SV):
    count = 0
    if (np.size(global_SV, axis=0) >= np.size(dis_SV, axis=0)):
        for i in range(np.size(dis_SV, axis=0)):
            if (not array_compare(dis_SV[i], global_SV)):
                count = count + 1
        count = count + np.size(global_SV, axis=0) - np.size(dis_SV, axis=0)
    else:
        for i in range(np.size(global_SV, axis=0)):
            if (not array_compare(global_SV[i], dis_SV)):
                count = count + 1
        count = count + np.size(dis_SV, axis=0) - np.size(global_SV, axis=0)
    return count


def training_iteration(Edge_node_n, Edge_data, Edge_label, Upload_support_vector_,
                       Collect_support_vector_, Collect_label, C, gamma):
    End_tag = False
    test = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    test.fit(Collect_support_vector_, Collect_label)
    # SVM_plot(Collect_support_vector_[:, 0] , Collect_support_vector_[:, 1] , Collect_label, test)
    print(test.score(Test_data, Test_label))

    broadcast_data = test.support_vectors_
    broadcast_label = Collect_label[test.support_]

    new_support_vector_plus = []
    new_support_plus = []
    new_support_vector_minus = []
    new_support_minus = []

    # Local model update
    for i in range(Edge_node_n):
        for j in range(np.size(Collect_support_vector_, axis=0)):
            if (not array_compare(Collect_support_vector_[j], Edge_data[i])):
                Edge_data[i] = np.concatenate((Edge_data[i], [Collect_support_vector_[j]]), axis=0)
                Edge_label[i] = np.concatenate((Edge_label[i], [Collect_label[j]]), axis=0)

        local_model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        local_model.fit(Edge_data[i], Edge_label[i])

        support_, support_vectors_, n_support_ = Ed.local_support(local_model)

        D_plus = np.multiply(Edge_label[i][support_[n_support_[0]:]],
                             test.decision_function(support_vectors_[n_support_[0]:, :]))
        D_minus = np.multiply(Edge_label[i][support_[:n_support_[0]]],
                              test.decision_function(support_vectors_[:n_support_[0], :]))

        S_plus_, SV_plus, _ \
            = Ed.upload_sort(support_[n_support_[0]:], support_vectors_[n_support_[0]:, :], D_plus)

        S_minus_, SV_minus, _ \
            = Ed.upload_sort(support_[:n_support_[0]], support_vectors_[:n_support_[0], :], D_minus)

        Distance_plus[i] = D_plus
        Distance_minus[i] = D_minus
        support_plus_[i] = S_plus_
        support_minus_[i] = S_minus_
        support_vectors_plus[i] = SV_plus
        support_vectors_minus[i] = SV_minus

        new_SV_plus = []
        new_SV_minus = []
        new_S_plus_ = []
        new_S_minus_ = []
        for j in range(np.size(SV_plus, axis=0)):
            if (not array_compare(SV_plus[j], Collect_support_vector_)):
                new_SV_plus.append(SV_plus[j])
                new_S_plus_.append(S_plus_[j])
        for j in range(np.size(SV_minus, axis=0)):
            if (not array_compare(SV_minus[j], Collect_support_vector_)):
                new_SV_minus.append(SV_minus[j])
                new_S_minus_.append(S_minus_[j])

        new_support_vector_plus.append(np.array(new_SV_plus))
        new_support_plus.append(np.array(new_S_plus_))
        new_support_vector_minus.append(np.array(new_SV_minus))
        new_support_minus.append(np.array(new_S_minus_))
        # print("Edge_data[%i].size is %s and Edge_label[%i].size is %d" % (
        # i, np.size(Edge_data[i], axis=0), i, np.size(Edge_label[i])))

    if((np.size(np.array(new_support_plus).reshape((-1)))==0)
            and (np.size(np.array(new_support_minus).reshape((-1)))==0)):
        End_tag = True

    stop_flag = [[False, False] for i in range(Edge_node_n)]
    while (1):
        for i in range(Edge_node_n):

            # look_plus = test.decision_function(new_support_vector_plus[i])
            # look_minus = test.decision_function(new_support_vector_minus[i])

            k1 = new_support_vector_plus[i][:1]
            k2 = new_support_vector_minus[i][:1]

            if (np.size(k1) == 0 and np.size(k2) == 0 or (stop_flag[i][0] and stop_flag[i][1])):
                stop_flag[i][0] = True
                stop_flag[i][1] = True
                continue
            elif (np.size(k1) == 0 and np.size(k2) != 0):
                stop_flag[i][0] = True
                if (test.decision_function(k2) * Edge_label[i][new_support_minus[i][:1]] > 1.0):
                    stop_flag[i][1] = True

                Upload_edge_support_vector_ = k2
                Upload_support_vector_[i] = \
                    np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_),axis=0)
                Upload_edge_label = Edge_label[i][new_support_minus[i][:1]]
                new_support_minus[i], new_support_vector_minus[i] \
                    = Ed.local_upload(new_support_minus[i], new_support_vector_minus[i])
                Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)

            elif (np.size(k1) != 0 and np.size(k2) == 0):
                stop_flag[i][1] = True
                if (test.decision_function(k1) * Edge_label[i][new_support_plus[i][:1]] > 1.0):
                    stop_flag[i][0] = True

                Upload_edge_support_vector_ = k1
                Upload_support_vector_[i] = \
                    np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_),axis=0)
                Upload_edge_label = Edge_label[i][new_support_plus[i][:1]]

                new_support_plus[i], new_support_vector_plus[i] = \
                    Ed.local_upload(new_support_plus[i], new_support_vector_plus[i])
                Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)

            else:
                if (test.decision_function(k1) * Edge_label[i][new_support_plus[i][:1]] > 1.0
                        and test.decision_function(k2) * Edge_label[i][new_support_minus[i][:1]] > 1.0):
                    stop_flag[i][0] = True
                    stop_flag[i][1] = True
                elif (test.decision_function(k1) * Edge_label[i][new_support_plus[i][:1]] > 1.0
                      and test.decision_function(k2) * Edge_label[i][new_support_minus[i][:1]] <= 1.0):
                    stop_flag[i][0] = True

                elif (test.decision_function(k1) * Edge_label[i][new_support_plus[i][:1]] <= 1.0
                      and test.decision_function(k2) * Edge_label[i][new_support_minus[i][:1]] > 1.0):
                    stop_flag[i][1] = True

                Upload_edge_support_vector_ = np.concatenate((k1, k2), axis=0)
                Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_),
                                                           axis=0)
                Upload_edge_label = np.concatenate(
                    (Edge_label[i][new_support_plus[i][:1]], Edge_label[i][new_support_minus[i][:1]]), axis=0)
                new_support_plus[i], new_support_vector_plus[i] \
                    = Ed.local_upload(new_support_plus[i], new_support_vector_plus[i])
                new_support_minus[i], new_support_vector_minus[i] \
                    = Ed.local_upload(new_support_minus[i], new_support_vector_minus[i])
                Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)

        Collect_support_vector_ = np.concatenate((Upload_support_vector_[0], Upload_support_vector_[1]))
        Collect_label = np.concatenate((Upload_label[0], Upload_label[1])).reshape((-1))

        if (Edge_node_n > 2):
            for j in range(2, Edge_node_n):
                Collect_support_vector_ = np.concatenate((Collect_support_vector_, Upload_support_vector_[j]))
                Collect_label = np.concatenate((Collect_label, Upload_label[j].reshape(-1)))

        Collect_support_vector_, Collect_label = shuffle(Collect_support_vector_, Collect_label)
        # old_support_vectors_ = test.support_vectors_

        test = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        test.fit(Collect_support_vector_, Collect_label)

        new_support_vectors_ = test.support_vectors_

        if (False not in np.array(stop_flag).reshape((-1))):
            break

    print(np.size(Collect_label, axis=0))
    print(test.n_support_)
    print(Ce.training_loss(test))

    return End_tag, new_support_vectors_, Upload_support_vector_, Collect_support_vector_, Collect_label


def training_iteration2(Edge_node_n, Edge_data, Edge_label, Upload_support_vector_,
                       Collect_support_vector_, Collect_label, C, gamma):
    End_tag = False
    test = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    test.fit(Collect_support_vector_, Collect_label)
    # SVM_plot(Collect_support_vector_[:, 0] , Collect_support_vector_[:, 1] , Collect_label, test)
    print(test.score(Test_data, Test_label))


    new_support_vector_plus = []
    new_support_plus = []
    new_support_vector_minus = []
    new_support_minus = []

    # Local model update
    for i in range(Edge_node_n):
        for j in range(np.size(Collect_support_vector_, axis=0)):
            if (not array_compare(Collect_support_vector_[j], Edge_data[i])):
                Edge_data[i] = np.concatenate((Edge_data[i], [Collect_support_vector_[j]]), axis=0)
                Edge_label[i] = np.concatenate((Edge_label[i], [Collect_label[j]]), axis=0)

        local_model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        local_model.fit(Edge_data[i], Edge_label[i])

        support_, support_vectors_, n_support_ = Ed.local_support(local_model)

        D_plus = np.multiply(Edge_label[i][support_[n_support_[0]:]],
                             test.decision_function(support_vectors_[n_support_[0]:, :]))
        D_minus = np.multiply(Edge_label[i][support_[:n_support_[0]]],
                              test.decision_function(support_vectors_[:n_support_[0], :]))

        S_plus_, SV_plus, D_order_plus \
            = Ed.upload_sort(support_[n_support_[0]:], support_vectors_[n_support_[0]:, :], D_plus)

        S_minus_, SV_minus, D_order_minus \
            = Ed.upload_sort(support_[:n_support_[0]], support_vectors_[:n_support_[0], :], D_minus)

        Distance_plus[i] = D_order_plus
        Distance_minus[i] = D_order_minus
        support_plus_[i] = S_plus_
        support_minus_[i] = S_minus_
        support_vectors_plus[i] = SV_plus
        support_vectors_minus[i] = SV_minus

        new_SV_plus = []
        new_SV_minus = []
        new_S_plus_ = []
        new_S_minus_ = []

        for j in range(np.size(SV_plus, axis=0)):
            if (not array_compare(SV_plus[j], Collect_support_vector_)):
                new_SV_plus.append(SV_plus[j])
                new_S_plus_.append(S_plus_[j])
        for j in range(np.size(SV_minus, axis=0)):
            if (not array_compare(SV_minus[j], Collect_support_vector_)):
                new_SV_minus.append(SV_minus[j])
                new_S_minus_.append(S_minus_[j])

        new_support_vector_plus.append(np.array(new_SV_plus))
        new_support_plus.append(np.array(new_S_plus_))
        new_support_vector_minus.append(np.array(new_SV_minus))
        new_support_minus.append(np.array(new_S_minus_))
        print("Edge_data[%i].size is %s and Edge_label[%i].size is %d" % (
            i, np.size(Edge_data[i], axis=0), i, np.size(Edge_label[i])))

    if ((np.size(np.array(new_support_plus).reshape((-1))) == 0)
            and (np.size(np.array(new_support_minus).reshape((-1))) == 0)):
        End_tag = True

    threshold = 1.01
    stop_flag = [[False, False] for i in range(Edge_node_n)]
    while (1):
        for i in range(Edge_node_n):
            k1 = new_support_vector_plus[i][:1]
            k2 = new_support_vector_minus[i][:1]

            if (np.size(k1) == 0 and np.size(k2) == 0 or (stop_flag[i][0] and stop_flag[i][1])):
                stop_flag[i][0] = True
                stop_flag[i][1] = True
                continue
            elif (np.size(k1) == 0 and np.size(k2) != 0):
                stop_flag[i][0] = True
                if(test.decision_function(k2)* Edge_label[i][new_support_minus[i][:1]] > threshold):
                    stop_flag[i][1] = True
                    continue
                else:
                    Upload_edge_support_vector_ = k2
                    Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_),
                                                               axis=0)

                    Upload_edge_label = Edge_label[i][new_support_minus[i][:1]]

                    new_support_minus[i], new_support_vector_minus[i] \
                        = Ed.local_upload(new_support_minus[i], new_support_vector_minus[i])
                    Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)

            elif (np.size(k1) != 0 and np.size(k2) == 0):
                stop_flag[i][1] = True
                if (test.decision_function(k1) * Edge_label[i][new_support_plus[i][:1]] > threshold):
                    stop_flag[i][0] = True
                    continue
                else:
                    Upload_edge_support_vector_ = k1
                    Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_),
                                                               axis=0)

                    Upload_edge_label = Edge_label[i][new_support_plus[i][:1]]

                    new_support_plus[i], new_support_vector_plus[i] \
                        = Ed.local_upload(new_support_plus[i], new_support_vector_plus[i])
                    Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)

            else:
                if(test.decision_function(k1) * Edge_label[i][new_support_plus[i][:1]] > threshold
                        and test.decision_function(k2) * Edge_label[i][new_support_minus[i][:1]] > threshold):
                    stop_flag[i][0] = True
                    stop_flag[i][1] = True
                    continue
                elif (test.decision_function(k1) * Edge_label[i][new_support_plus[i][:1]] > threshold
                        and test.decision_function(k2) * Edge_label[i][new_support_minus[i][:1]] <= threshold):
                    stop_flag[i][0] = True
                    Upload_edge_support_vector_ = k2
                    Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_),
                                                               axis=0)
                    Upload_edge_label = Edge_label[i][new_support_minus[i][:1]]

                    new_support_minus[i], new_support_vector_minus[i] \
                        = Ed.local_upload(new_support_minus[i], new_support_vector_minus[i])
                    Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)

                elif(test.decision_function(k1) * Edge_label[i][new_support_plus[i][:1]] <= threshold
                        and test.decision_function(k2) * Edge_label[i][new_support_minus[i][:1]] > threshold):
                    stop_flag[i][1] = True
                    Upload_edge_support_vector_ = k1
                    Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_),
                                                               axis=0)

                    Upload_edge_label = Edge_label[i][new_support_plus[i][:1]]

                    new_support_plus[i], new_support_vector_plus[i] \
                        = Ed.local_upload(new_support_plus[i], new_support_vector_plus[i])
                    Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)

                elif(test.decision_function(k1) * Edge_label[i][new_support_plus[i][:1]] <= threshold
                        and test.decision_function(k2) * Edge_label[i][new_support_minus[i][:1]] <= threshold):
                    Upload_edge_support_vector_ = np.concatenate((k1, k2), axis=0)
                    Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_),
                                                               axis=0)

                    Upload_edge_label = np.concatenate(
                        (Edge_label[i][new_support_plus[i][:1]], Edge_label[i][new_support_minus[i][:1]]), axis=0)

                    new_support_plus[i], new_support_vector_plus[i] \
                        = Ed.local_upload(new_support_plus[i], new_support_vector_plus[i])
                    new_support_minus[i], new_support_vector_minus[i] \
                        = Ed.local_upload(new_support_minus[i], new_support_vector_minus[i])

                    Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)

        Collect_support_vector_ = np.concatenate((Upload_support_vector_[0], Upload_support_vector_[1]))
        Collect_label = np.concatenate((Upload_label[0], Upload_label[1])).reshape((-1))

        if (Edge_node_n > 2):
            for j in range(2, Edge_node_n):
                Collect_support_vector_ = np.concatenate((Collect_support_vector_, Upload_support_vector_[j]))
                Collect_label = np.concatenate((Collect_label, Upload_label[j].reshape(-1)))

        Collect_support_vector_, Collect_label = shuffle(Collect_support_vector_, Collect_label)
        # old_support_vectors_ = test.support_vectors_

        test = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        test.fit(Collect_support_vector_, Collect_label)

        new_support_vectors_ = test.support_vectors_

        if (False not in np.array(stop_flag).reshape((-1))):
            break

    print(np.size(Collect_label, axis=0))
    print(test.n_support_)
    print(Ce.training_loss(test))

    return End_tag, new_support_vectors_, Upload_support_vector_, Collect_support_vector_, Collect_label


# Train_data, Train_label, Test_data, Test_label = read_UCI_data(1)
Train_data, Train_label, Test_data, Test_label = read_skin()

print("global size is %s" % (np.size(Train_label)))


# Training_set = sklearn.datasets.load_svmlight_file('german')
#
# Train_label_t = Training_set[1]
# Train_data_t = np.array(Training_set[0].todense())
# Train_data_t = preprocessing.normalize(Train_data_t)
#
# Train_data_s,Train_label_s = shuffle(Train_data_t, Train_label_t)
#
# Train_label = Train_label_s
# Train_data = Train_data_s


C = 3.0
gamma = 0.3

Global_model = svm.SVC(C=C, kernel='rbf', gamma=gamma)

Global_model.fit(Train_data, Train_label)
print(Ce.training_loss(Global_model))

# print (np.size(Global_model.dual_coef_[0]))
# SVM_plot(Train_data[:, 0], Train_data[:, 1], Train_label, Global_model)

Only_SV_model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
data, label = shuffle(Global_model.support_vectors_, Train_label[Global_model.support_])
# Only_SV_model.fit(Global_model.support_vectors_, Train_label[Global_model.support_])
Only_SV_model.fit(data, label)
print(Ce.training_loss(Only_SV_model))


print(Global_model.n_support_)
print(Only_SV_model.n_support_)

# # SVM_plot(Train_data[:, 0], Train_data[:, 1], Train_label, Global_model)
# # SVM_plot(Global_model.support_vectors_[:, 0], Global_model.support_vectors_[:, 1], Train_label[Global_model.support_], Only_SV_model)

Edge_node_n = 10
Edge_data, Edge_label, Global_index = data_partition(Train_data, Train_label, Edge_node_n)
# Edge_data, Edge_label, Global_index = kmeans_partition(Train_data, Train_label, Edge_node_n)
# Edge_data, Edge_label, Global_index = k_means_random_partition(Train_data, Train_label, Edge_node_n, 5, True)

Edge_data_all = copy.deepcopy(Edge_data)
Edge_label_all = copy.deepcopy(Edge_label)


for i in range(Edge_node_n):
    print("Edge_data[%i].size is %s and Edge_label[%i].size is %d" % (
    i, np.size(Edge_data[i], axis=0), i, np.size(Edge_label[i])))

Distance_plus = []
Distance_minus = []
support_plus_ = []
support_minus_ = []
support_vectors_plus = []
support_vectors_minus = []

for i in range(Edge_node_n):
    local_model = Ed.local_train(Edge_data[i], Edge_label[i], C, gamma, 'rbf')

    # # SVM_plot(Edge_data[i][:, 0], Edge_data[i][:, 1], Edge_label[i], local_model)
    support_, support_vectors_, n_support_ = Ed.local_support(local_model)

    print("i = %i, n_support = %s" % (i, np.sum(n_support_)))

    D_plus = np.multiply(Edge_label[i][support_[n_support_[0]:]],
                         local_model.decision_function(support_vectors_[n_support_[0]:, :]))
    D_minus = np.multiply(Edge_label[i][support_[:n_support_[0]]],
                          local_model.decision_function(support_vectors_[:n_support_[0], :]))

    S_plus_, SV_plus, _ \
        = Ed.upload_sort(support_[n_support_[0]:], support_vectors_[n_support_[0]:, :], D_plus)

    S_minus_, SV_minus, _ \
        = Ed.upload_sort(support_[:n_support_[0]], support_vectors_[:n_support_[0], :], D_minus)

    Distance_plus.append(D_plus)
    Distance_minus.append(D_minus)
    support_plus_.append(S_plus_)
    support_minus_.append(S_minus_)
    support_vectors_plus.append(SV_plus)
    support_vectors_minus.append(SV_minus)

'''
 If uploading all the local support vectors
'''

local_support_vector = np.concatenate((support_vectors_plus[0], support_vectors_minus[0]), axis=0)
local_label_plus = np.ones(np.size(support_vectors_plus[0], axis=0))
local_label_minus = np.ones(np.size(support_vectors_minus[0], axis=0)) * (-1)
local_label = np.concatenate((local_label_plus, local_label_minus), axis=0)
for i in range(1, Edge_node_n):
    local_support_vector = np.concatenate((local_support_vector, support_vectors_plus[i], support_vectors_minus[i]),
                                          axis=0)
    local_label_plus = np.ones(np.size(support_vectors_plus[i], axis=0))
    local_label_minus = np.ones(np.size(support_vectors_minus[i], axis=0)) * (-1)
    local_label = np.concatenate((local_label, local_label_plus, local_label_minus), axis=0)

All_upload_model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
All_upload_model.fit(local_support_vector, local_label)

# # SVM_plot(local_support_vector[:, 0], local_support_vector[:, 1], local_label, All_upload_model)
print(np.size(local_label, axis=0))
print(All_upload_model.n_support_)
'''
Central update
Outlier upload
'''

Updated_support_plus_ = []
Updated_support_minus_ = []
Updated_support_vectors_plus = []
Updated_support_vectors_minus = []

Upload_support_vector_ = []
Upload_label = []

##the t part is added to make it consist
Upload_support_vector_t = []
Upload_label_t = []

# Upload outliers
Outlier_plus = np.zeros((Edge_node_n, 1), dtype=np.int32)
Outlier_minus = np.zeros((Edge_node_n, 1), dtype=np.int32)
#####################################################################
# Outlier_coef = np.zeros((Edge_node_n,2), dtype= float)
# print(Outlier_coef)

# base_coef = 2

# for i in range(Edge_node_n):
#     Outlier_plus[i] = np.sum((Distance_plus[i]<=0))
#     Outlier_minus[i] = np.sum((Distance_minus[i]<=0))
#     min = np.min([Outlier_plus[i], Outlier_minus[i]])
#     if (min == 0):
#         Outlier_coef[i][0] = base_coef
#         Outlier_coef[i][1] = base_coef
#     elif (min == Outlier_plus[i]):
#         Outlier_coef[i][0] = base_coef
#         Outlier_coef[i][1] = base_coef * min / Outlier_minus[i]
#     else:
#         Outlier_coef[i][1] = base_coef
#         Outlier_coef[i][0] = base_coef * min / Outlier_plus[i]


# for i in range(Edge_node_n):
#     Outlier_plus[i] = np.sum((Distance_plus[i]<=0))
#     # Outlier_minus[i] = np.sum((Distance_minus[i]<=0))

#     Outlier_n = np.max([Outlier_plus[i], Outlier_plus[i]])

#     Upload_edge_support_vector_ = (support_vectors_plus[i][: (int(Outlier_coef[i][0]*Outlier_n ))   ,:])

#     if (i > 0):
#         Upload_support_vector_t = np.concatenate ( (Upload_support_vector_t,  Upload_edge_support_vector_), axis = 0)
#     else:
#         Upload_support_vector_t = Upload_edge_support_vector_

#     Upload_support_vector_.append (Upload_edge_support_vector_)

#     # print ("i = %s" %(i))
#     # print (Upload_edge_support_vector_)
#     # print (Upload_support_vector_)

#     label_plus = Edge_label[i][support_plus_[i][:   (int(Outlier_coef[i][0]*Outlier_n ))  ]]
#     Upload_edge_label = (label_plus)

#     if (i > 0):
#         Upload_label_t= np.concatenate ( (Upload_label_t,  Upload_edge_label), axis = 0)
#     else:
#         Upload_label_t= Upload_edge_label

#     Upload_label.append(Upload_edge_label)


#     # print (Upload_label)

#     US_plus_, USV_plus\
#         = Ed.local_upload_outlier(support_plus_[i], support_vectors_plus[i], 2*Outlier_n)

#     Updated_support_plus_.append(US_plus_)
#     Updated_support_vectors_plus.append(USV_plus)


# for i in range(Edge_node_n):
#     # Outlier_plus[i] = np.sum((Distance_plus[i]<=0))
#     Outlier_minus[i] = np.sum((Distance_minus[i]<=0))

#     Outlier_n = np.max([Outlier_minus[i], Outlier_minus[i]])

#     Upload_edge_support_vector_ = (support_vectors_minus[i][:   (int(Outlier_coef[i][1]*Outlier_n ))   ,:])

#     Upload_support_vector_t = np.concatenate ( (Upload_support_vector_t,  Upload_edge_support_vector_), axis = 0)
#     Upload_support_vector_.append (Upload_edge_support_vector_)


#     # print ("i = %s" %(i))
#     # print (Upload_edge_support_vector_)
#     # print (Upload_support_vector_)

#     label_minus = Edge_label[i][support_minus_[i][:   (int(Outlier_coef[i][1]*Outlier_n ))  ]]
#     Upload_edge_label = (label_minus)

#     Upload_label_t= np.concatenate ( (Upload_label_t,  Upload_edge_label), axis = 0)
#     Upload_label.append(Upload_edge_label)


#     US_minus_, USV_minus\
#         = Ed.local_upload_outlier(support_minus_[i], support_vectors_minus[i], 2*Outlier_n)

#     Updated_support_minus_.append(US_minus_)
#     Updated_support_vectors_minus.append(USV_minus)

# # print (Upload_label)

# Collect_support_vector_ = Upload_support_vector_t
# Collect_label = Upload_label_t

# # make this to be continusous (Upload_Label has 20 elements, but before iteration, it should be in size of 10)
# T_label = []
# T_sv = []

# for i in range(Edge_node_n):
#     T1 = np.concatenate( (Upload_label[i], Upload_label[i + Edge_node_n]), axis = 0  )
#     T2 = np.concatenate ( (Upload_support_vector_[i], Upload_support_vector_[i + Edge_node_n]), axis = 0  )

#     T_label.append (  T1   )
#     T_sv.append( T2 )

# Upload_label = T_label
# Upload_support_vector_ = T_sv
#######################################################################################
# base_coef = 1
# for i in range(Edge_node_n):
#     Outlier_plus[i] = np.sum((Distance_plus[i]<=0))
#     Outlier_minus[i] = np.sum((Distance_minus[i]<=0))
#     print ("Outlier_plus[%i] = %s" %(i, Outlier_plus[i][0]))
#     print ("Outlier_minus[%i] = %s" %(i, Outlier_minus[i][0] ))

#     label_plus = Edge_label[i][support_plus_[i][:(int(base_coef*Outlier_plus[i][0]))]]
#     label_minus = Edge_label[i][support_minus_[i][:(int(base_coef*Outlier_minus[i][0]))]]

#     Upload_edge_support_vector_ = np.concatenate(
#         (support_vectors_plus[i][:(int(base_coef*Outlier_plus[i][0])),:],support_vectors_minus[i][:(int(base_coef*Outlier_minus[i][0])),:]), axis = 0)
#     Upload_edge_label = np.concatenate((label_plus, label_minus))

#     Upload_support_vector_.append(Upload_edge_support_vector_)
#     Upload_label.append(Upload_edge_label)

#     US_plus_, USV_plus\
#         = Ed.local_upload_outlier(support_plus_[i], support_vectors_plus[i], (int(base_coef*Outlier_plus[i][0])) )
#     US_minus_, USV_minus\
#         = Ed.local_upload_outlier(support_minus_[i], support_vectors_minus[i], (int(base_coef*Outlier_minus[i][0])) )

#     Updated_support_plus_.append(US_plus_)
#     Updated_support_minus_.append(US_minus_)
#     Updated_support_vectors_plus.append(USV_plus)
#     Updated_support_vectors_minus.append(USV_minus)
#     # print ("i = %s" %(i))
#     # print (Updated_support_vectors_minus)

# Collect_support_vector_ = np.concatenate((Upload_support_vector_[0], Upload_support_vector_[1]))
# Collect_label = np.concatenate((Upload_label[0], Upload_label[1])).reshape((-1))\


# if (Edge_node_n > 2):
#     for j in range(2, Edge_node_n):
#         Collect_support_vector_ = np.concatenate((Collect_support_vector_, Upload_support_vector_[j]))
#         Collect_label = np.concatenate((Collect_label, Upload_label[j].reshape(-1)))

# num_out = np.size(Collect_label)
# print ("num_out = %s" %(num_out))

# for i in range(Edge_node_n):
#     Outlier_n = np.max([Outlier_plus[i],Outlier_minus[i] ])
#     if (Outlier_plus[i][0] == Outlier_minus[i][0]):
#         continue
#     elif (Outlier_plus[i][0] > Outlier_minus[i][0]):
#         label_minus = Edge_label[i][Updated_support_minus_[i][:(int(base_coef*Outlier_n) - Outlier_minus[i][0]  ) ]]
#         Upload_edge_label = label_minus
#         Upload_label[i] = np.concatenate (  (Upload_label[i], label_minus), axis = 0 )

#         SV_minus = Updated_support_vectors_minus[i][:(int(base_coef*Outlier_n) - Outlier_minus[i][0]  ),:]
#         Upload_edge_support_vector_  = SV_minus
#         Upload_support_vector_[i] = np.concatenate (  (Upload_support_vector_[i], SV_minus) )
#         US_minus_, USV_minus\
#             = Ed.local_upload_outlier(Updated_support_minus_[i], Updated_support_vectors_minus[i], (int(base_coef*Outlier_n) - Outlier_minus[i][0]  ) )
#         Updated_support_minus_[i] = (US_minus_)
#         Updated_support_vectors_minus[i] = (USV_minus)
#     else:
#         label_plus = Edge_label[i][Updated_support_plus_[i][:(int(base_coef*Outlier_n) - Outlier_plus[i][0]  ) ]]
#         Upload_edge_label = label_plus
#         Upload_label[i] = np.concatenate (  (Upload_label[i], label_plus), axis = 0 )
#         SV_plus = Updated_support_vectors_plus[i][:(int(base_coef*Outlier_n) - Outlier_plus[i][0]  ),:]
#         Upload_edge_support_vector_ = SV_plus
#         Upload_support_vector_[i] = np.concatenate (  (Upload_support_vector_[i], SV_plus) )
#         US_plus_, USV_plus\
#             = Ed.local_upload_outlier(Updated_support_plus_[i], Updated_support_vectors_plus[i], (int(base_coef*Outlier_n) - Outlier_plus[i][0]  ) )
#         Updated_support_plus_[i] = (US_plus_)
#         Updated_support_vectors_plus[i] = (USV_plus)

#     Collect_label = np.concatenate((Collect_label, Upload_edge_label), axis = 0)
#     Collect_support_vector_ = np.concatenate((Collect_support_vector_, Upload_edge_support_vector_), axis = 0)
#     # print (np.size (Collect_label))
#     # print (np.size ( Collect_support_vector_, axis = 0))
############################################# outlier upload and balance part
stop_flag = True
ite_count = 0
Upper_bound = Ce.upper_bound(Edge_data, Edge_label, C, gamma, Edge_node_n)

base_coef = 1
for i in range(Edge_node_n):
    Outlier_plus[i] = 1
    Outlier_minus[i] = 1

    label_plus = Edge_label[i][support_plus_[i][:(int(base_coef * Outlier_plus[i][0]))]]
    label_minus = Edge_label[i][support_minus_[i][:(int(base_coef * Outlier_minus[i][0]))]]

    Upload_edge_support_vector_ = np.concatenate(
        (support_vectors_plus[i][:(int(base_coef * Outlier_plus[i][0])), :],
         support_vectors_minus[i][:(int(base_coef * Outlier_minus[i][0])), :]), axis=0)
    Upload_edge_label = np.concatenate((label_plus, label_minus))

    Upload_support_vector_.append(Upload_edge_support_vector_)
    Upload_label.append(Upload_edge_label)

    US_plus_, USV_plus \
        = Ed.local_upload_outlier(support_plus_[i], support_vectors_plus[i], (int(base_coef * Outlier_plus[i][0])))
    US_minus_, USV_minus \
        = Ed.local_upload_outlier(support_minus_[i], support_vectors_minus[i], (int(base_coef * Outlier_minus[i][0])))

    Updated_support_plus_.append(US_plus_)
    Updated_support_minus_.append(US_minus_)
    Updated_support_vectors_plus.append(USV_plus)
    Updated_support_vectors_minus.append(USV_minus)
    # print ("i = %s" %(i))
    # print (Updated_support_vectors_minus)

Collect_support_vector_ = np.concatenate((Upload_support_vector_[0], Upload_support_vector_[1]))
Collect_label = np.concatenate((Upload_label[0], Upload_label[1])).reshape((-1))
if (Edge_node_n > 2):
    for j in range(2, Edge_node_n):
        Collect_support_vector_ = np.concatenate((Collect_support_vector_, Upload_support_vector_[j]))
        Collect_label = np.concatenate((Collect_label, Upload_label[j].reshape(-1)))

while (1):
    print("ite_count = %s" % (ite_count))
    ite_count += 1
    test = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    test.fit(Collect_support_vector_, Collect_label)
    support_ = test.support_
    # break
    # if (ite_count == 2):
    #     break

    # print (np.size (test.dual_coef_))
    # print (Collect_label)
    # print (test.dual_coef_[0:num_out])
    old_collect_sv = Collect_support_vector_
    stop_flag = True
    index = 0
    X = [-1, -1]
    y = 0

    central_training_loss = Ce.training_loss(test)
    if (central_training_loss > Upper_bound):
        stop_flag = False

    if (stop_flag):
        print("stop_flag == True")
        break

    # SVM_plot(Collect_support_vector_[:, 0] , Collect_support_vector_[:, 1] , Collect_label, test, X, y )

    for i in range(Edge_node_n):
        # print ("i = %i" %(i))
        # print ("Updated_support_vectors_plus[%i].size is %s" %(i, np.size(Updated_support_vectors_plus[i])/3) )
        # print ("Updated_support_vectors_minus[%i].size is %s" %(i, np.size(Updated_support_vectors_minus[i])/3) )

        break_flag = False
        if (np.size(Updated_support_vectors_minus[i]) == 0):
            if (np.size(Updated_support_vectors_plus[i]) == 0):
                break_flag = True
            else:
                label_plus = Edge_label[i][Updated_support_plus_[i][:1]]
                Upload_edge_label = label_plus
                Upload_edge_support_vector_ = np.array([Updated_support_vectors_plus[i][0]])
        elif (np.size(Updated_support_vectors_plus[i]) == 0):
            Upload_edge_support_vector_ = np.array([Updated_support_vectors_minus[i][0]])
            label_minus = Edge_label[i][Updated_support_minus_[i][:1]]
            Upload_edge_label = label_minus
        else:
            Upload_edge_support_vector_ = np.concatenate(
                (np.array([Updated_support_vectors_plus[i][0]]), np.array([Updated_support_vectors_minus[i][0]])),
                axis=0)
            label_plus = Edge_label[i][Updated_support_plus_[i][:1]]
            label_minus = Edge_label[i][Updated_support_minus_[i][:1]]
            Upload_edge_label = np.concatenate((label_plus, label_minus))

        if (break_flag):
            continue

        # print (Upload_edge_support_vector_)
        Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)
        print("Upload_label[%i].size is %s " % (i, np.size(Upload_label[i])))
        Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_), axis=0)

        US_plus_, USV_plus \
            = Ed.local_upload_outlier(Updated_support_plus_[i], Updated_support_vectors_plus[i], (1))
        # print ("Updated_support_plus_[%i] is %s " %(i, np.size (Updated_support_plus_[i]) ) )
        # print ("Updated_support_vectors_plus[%i] is %s"  %(i,np.size (Updated_support_vectors_plus[i])/2 ) )
        US_minus_, USV_minus \
            = Ed.local_upload_outlier(Updated_support_minus_[i], Updated_support_vectors_minus[i], (1))

        Updated_support_plus_[i] = (US_plus_)
        Updated_support_minus_[i] = (US_minus_)
        Updated_support_vectors_plus[i] = (USV_plus)
        Updated_support_vectors_minus[i] = (USV_minus)

        Collect_label = np.concatenate((Collect_label, Upload_edge_label), axis=0)
        Collect_support_vector_ = np.concatenate((Collect_support_vector_, Upload_edge_support_vector_), axis=0)
        # print ("Collect_support_vector.size is %s" %(np.size(Collect_support_vector_, axis = 0)))
    break_flag = True
    if (np.size(Collect_support_vector_) == np.size(old_collect_sv)):
        break_flag = True
    else:
        break_flag = False
    if (break_flag):
        print("Collect_sv = Old_collect_sv")
        break

    # Collect_support_vector_ = np.concatenate((Upload_support_vector_[0], Upload_support_vector_[1]))
    # Collect_label = np.concatenate((Upload_label[0], Upload_label[1])).reshape((-1))
    # if (Edge_node_n > 2):
    #     for j in range(2, Edge_node_n):
    #         Collect_support_vector_ = np.concatenate((Collect_support_vector_, Upload_support_vector_[j]))
    #         Collect_label = np.concatenate((Collect_label, Upload_label[j].reshape(-1)))
    # print (Collect_support_vector_)

print(np.size(Collect_label, axis=0))

# while((np.sum(test.n_support_)) == np.size(Collect_support_vector_, axis=0)):
# # while ((np.sum(test.n_support_)) == np.size(Collect_support_vector_, axis=0)):
#     for i in range(Edge_node_n):
#         k1= [Updated_support_vectors_plus[i][0]]
#         k2 = [Updated_support_vectors_minus[i][0]]
#         Upload_edge_support_vector_ = np.concatenate(
#             (k1, k2), axis=0)
#         Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_), axis=0)
#
#         Upload_edge_label = [1.0, -1.0]
#         Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)
#
#         Updated_support_plus_[i], Updated_support_vectors_plus[i] \
#             = Ed.local_upload(Updated_support_plus_[i], Updated_support_vectors_plus[i])
#         Updated_support_minus_[i], Updated_support_vectors_minus[i] \
#             = Ed.local_upload(Updated_support_minus_[i], Updated_support_vectors_minus[i])
#
#     Collect_support_vector_ = np.concatenate((Upload_support_vector_[0], Upload_support_vector_[1]))
#     Collect_label = np.concatenate((Upload_label[0], Upload_label[1])).reshape((-1))
#
#     if(Edge_node_n > 2):
#         for j in range(2,Edge_node_n):
#             Collect_support_vector_ = np.concatenate((Collect_support_vector_,Upload_support_vector_[j]))
#             Collect_label = np.concatenate((Collect_label, Upload_label[j].reshape(-1)))
#     test = svm.SVC(C=C, kernel='rbf', gamma=gamma)
#     test.fit(Collect_support_vector_, Collect_label)
#
#
# print(np.size(Collect_label, axis=0))
# print(test.n_support_)
ite = 0
End_flag, new_support_vectors_, Upload_support_vector_, Collect_support_vector_, Collect_label \
    = training_iteration(Edge_node_n,
                         Edge_data,
                         Edge_label,
                         Upload_support_vector_,
                         Collect_support_vector_,
                         Collect_label,
                         C, gamma)
print("ite = %s" % (ite))
ite += 1
while (not End_flag):
    End_flag, new_support_vectors_, Upload_support_vector_, Collect_support_vector_, Collect_label \
        = training_iteration(
        Edge_node_n, Edge_data, Edge_label, Upload_support_vector_,
        Collect_support_vector_, Collect_label, C, gamma)
    print("ite = %s" % (ite))
    ite += 1

print(SV_diff_count(Global_model.support_vectors_, new_support_vectors_))








'''
 Uploading all the local support vectors:
'''

print("For uploading all the local support vectors:")

Edge_num = np.zeros(Edge_node_n).tolist()

for i in range(Edge_node_n):
    Edge_num[i] = np.size(Edge_label_all[i])

Distance_plus = []
Distance_minus = []
support_plus_ = []
support_minus_ = []
support_vectors_plus = []
support_vectors_minus = []

for i in range(Edge_node_n):
    local_model = Ed.local_train(Edge_data_all[i], Edge_label_all[i], C, gamma, 'rbf')

    # # SVM_plot(Edge_data[i][:, 0], Edge_data[i][:, 1], Edge_label[i], local_model)
    support_, support_vectors_, n_support_ = Ed.local_support(local_model)

    support_vectors_plus.append(support_vectors_[n_support_[0]:, :])
    support_vectors_minus.append(support_vectors_[:n_support_[0], :])

    # # SVM_plot(Edge_data[i][:, 0], Edge_data[i][:, 1], Edge_label[i], local_model)

local_support_vector = np.concatenate((support_vectors_plus[0], support_vectors_minus[0]), axis=0)
local_label_plus = np.ones(np.size(support_vectors_plus[0], axis=0))
local_label_minus = np.ones(np.size(support_vectors_minus[0], axis=0)) * (-1)
local_label = np.concatenate((local_label_plus, local_label_minus), axis=0)
for i in range(1, Edge_node_n):
    local_support_vector = np.concatenate((local_support_vector, support_vectors_plus[i], support_vectors_minus[i]),
                                          axis=0)
    local_label_plus = np.ones(np.size(support_vectors_plus[i], axis=0))
    local_label_minus = np.ones(np.size(support_vectors_minus[i], axis=0)) * (-1)
    local_label = np.concatenate((local_label, local_label_plus, local_label_minus), axis=0)

old_support_vectors_ = []

central_model, global_support_vector, global_label = \
    Ce.central_training(local_support_vector, local_label, C, gamma, 'rbf')

new_support_vectors_ = central_model.support_vectors_
# SVM_plot(local_support_vector[:, 0], local_support_vector[:, 1], local_label, All_upload_model)
print(np.size(local_label, axis=0))
print(central_model.n_support_)

ite_count = 1

remain_SV = np.copy(local_support_vector)

while (not Ed.SV_compare(old_support_vectors_, new_support_vectors_)):
    ite_count += 1

    new_support_vector_plus = []
    new_support_vector_minus = []
    print("global_support_v.size is %s" % (np.size(global_support_vector, axis=0)))

    # delete the additional local vector before receive the central node delivery
    for i in range(Edge_node_n):
        # index = 0
        # # print ("i is %i" %(i))
        # before_size = np.size (Edge_data[i], axis= 0)
        # # print (before_size)
        # for j in Edge_data[i]:
        #     for k in global_support_vector:
        #         # print (j)
        #         if (vector_compare(k,j)):
        #             Edge_data[i] = np.delete(Edge_data[i], index, 0)
        #             Edge_label[i] = np.delete(Edge_label[i], index, 0)
        #             index -= 1
        #     index +=1
        #
        # # print(np.size (Edge_data[i], axis=0))
        # num_delete += before_size -  np.size (Edge_data[i], axis= 0)

        Edge_data_all[i], Edge_label_all[i] = \
            Ed.data_mix(Edge_data_all[i], Edge_label_all[i], global_support_vector, global_label)

        # print ("Edge_label[%i].size after receiving the delivered data is %s" %(i,np.size(Edge_label_all[i] )))

        local_model = Ed.local_train(Edge_data_all[i], Edge_label_all[i], C, gamma, 'rbf')
        support_, support_vectors_, n_support_ = Ed.local_support(local_model)

        # print(n_support_)
        # delete the additional local support vector
        # index = 0
        # for j in support_vectors_:
        #     for k in support_vectors_plus:
        #         if (vector_compare(k,j)):
        #             support_vectors_ = np.delete(support_vectors_, index, 0)
        #             n_support_[1] -= 1
        #             index -= 1
        #     for k in support_vectors_minus:
        #         if (vector_compare(k,j)):
        #             support_vectors_ = np.delete(support_vectors_, index, 0)
        #             n_support_[0] -= 1
        #             index -= 1
        #     index += 1
        for j in support_vectors_[n_support_[0]:, :]:
            if (not Ed.array_compare(j, remain_SV)):
                new_support_vector_plus.append(j)

        for j in support_vectors_[:n_support_[0], :]:
            if (not Ed.array_compare(j, remain_SV)):
                new_support_vector_minus.append(j)
        # print("support_v_p.size is %a " %( support_vectors_plus ))
        # print("(support_vectors_[n_support_[0]:, :]) is %a " %( support_vectors_[n_support_[0]:, :]))

    new_support_vector_plus = np.array(new_support_vector_plus)
    new_support_vector_minus = np.array(new_support_vector_minus)
    print("num_plus is %s " %(np.size(new_support_vector_plus, axis=0)))
    print("num_minus is %s " %(np.size(new_support_vector_minus, axis=0)))
    '''
    Uploading all the local support vectors
    '''
    # print (np.size(support_vectors_plus, axis = 0))
    # print (support_vectors_plus)

    if (np.size(new_support_vector_plus, axis=0) != 0 and np.size(new_support_vector_minus, axis=0) != 0):
        local_support_vector = \
            np.concatenate((local_support_vector, new_support_vector_plus, new_support_vector_minus), axis=0)
        local_label_plus = np.ones(np.size(new_support_vector_plus, axis=0))
        local_label_minus = np.ones(np.size(new_support_vector_minus, axis=0)) * (-1)
        local_label = np.concatenate((local_label, local_label_plus, local_label_minus), axis=0)
    elif (np.size(new_support_vector_plus, axis=0) == 0 and np.size(new_support_vector_minus, axis=0) != 0):
        local_support_vector = \
            np.concatenate((local_support_vector, new_support_vector_minus), axis=0)
        local_label_minus = np.ones(np.size(new_support_vector_minus, axis=0)) * (-1)
        local_label = np.concatenate((local_label, local_label_minus), axis=0)
    elif (np.size(new_support_vector_plus, axis=0) != 0 and np.size(new_support_vector_minus, axis=0) == 0):
        local_support_vector = \
            np.concatenate((local_support_vector, new_support_vector_plus), axis=0)
        local_label_plus = np.ones(np.size(new_support_vector_plus, axis=0))
        local_label = np.concatenate((local_label, local_label_plus), axis=0)

    # for i in range(1,Edge_node_n):
    #     local_support_vector = np.concatenate((local_support_vector, support_vectors_plus[i], support_vectors_minus[i]), axis=0)
    #     local_label_plus = np.ones(np.size(support_vectors_plus[i], axis=0))
    #     local_label_minus = np.ones(np.size(support_vectors_minus[i], axis=0)) * (-1)
    #     local_label = np.concatenate((local_label, local_label_plus, local_label_minus), axis=0)

    # print ("local_sv.size is %s" %(np.size(local_support_vector, axis = 0)))
    # print ("(after delete) (remain_SV.size) is %s " % (np.size(remain_SV,axis=0)))
    remain_SV = np.copy(local_support_vector)
    print("The overall transformed sv num (remain_SV.size) is %s " % (np.size(remain_SV, axis=0)))

    old_support_vectors_ = central_model.support_vectors_

    central_model, global_support_vector, global_label = \
        Ce.central_training(local_support_vector, local_label, C, gamma, 'rbf')

    new_support_vectors_ = central_model.support_vectors_
    print(Ce.training_loss(central_model))
    print(np.size(local_label, axis=0))
    print(central_model.n_support_)

# SVM_plot(local_support_vector[:, 0], local_support_vector[:, 1], local_label, central_model)

print("ite_count = %i" % (ite_count))
# for i in range(Edge_node_n):
#     print("the net addition in node %i is %s" % (i, np.size(Edge_label_all[i]) - Edge_num[i]))
print("The overall transformed sv num (remain_SV.size) is %s " % (np.size(remain_SV, axis=0)))
print(SV_diff_count(Global_model.support_vectors_, new_support_vectors_))


