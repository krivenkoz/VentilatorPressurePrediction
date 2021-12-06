"""
                Helper functions for sktime, dl4tsc, lstmfcn frameworks

get_accuracy_classic_tsc - ready
convert2rnn - ready
convert2cnn - ready
convert2sktime - ready
read_dataset_lstmfcn_split2val_reduced - ready
read_dataset_lstmfcn_reduced - ready
read_dataset_lstmfcn_split2val - ready
read_dataset_lstmfcn - ready
class_activation_map - ready
get_accuracy - ready
create_nn_classifier - ready
fit_nn_classifier - ready
read_train_dataset_dl4tsc_split2val_reduced - ready
read_train_dataset_dl4tsc_split2val - ready
read_dataset_dl4tsc_reduced - ready
read_dataset_dl4tsc - ready
znorm - ready
get_all_instances_sktime_format - ready
get_instance_sktime_format - ready
set_instance_sktime_format - ready

Version: 1.0
Status: ready
2020/06/19
By Krivenko S.S.
"""
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sktime.utils.load_data import load_from_tsfile_to_dataframe

# -------------------------------- [ get_accuracy_classic_tsc ] -------------------------------- #
def get_accuracy_classic_tsc(y_true, y_pred):
    """
    Calc sensitivity, specificity, accuracy

    Args:
        y_true: grounf truth
        y_pred: predicted values

    Returns:
        classic_metrics: dict with sensitivity, specificity, accuracy        
    """
    classic_metrics = dict()

    false_positive_rate, recall, thresholds = roc_curve(y_true.astype(int), y_pred.astype(int))
    roc_auc = auc(false_positive_rate, recall)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    classic_metrics['sensitivity'] = tp / (tp + fn)
    classic_metrics['specificity'] = tn / (tn + fp)
    classic_metrics['accuracy'] = (tn + tp) / (tn + tp + fn + fp)
    classic_metrics['AUC'] = roc_auc
    return classic_metrics
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ convert2rnn ] -------------------------------- #
def convert2rnn(sdf, target, ts_length, ds_name, sdf_test=None, split_flag=1, znormal=False):
    """
    Convert ECG/RR dataframe to RNN dataframe

    Args:
        sdf: standard dataframe with train data
        target: target name in sdf
        ts_length: time series length
        ds_name: name of dataset
        split_flag: 0 - no splitting, 1 - split to train and test, 2 - read sdf_test
        znormal: Z-normalization
        sdf_test: standard dataframe with test data

    Returns:
        ts_X: DNN dataframe with X
        ts_y: ndarray with y
    """
    y_full = sdf[target].values.astype(int)
    sdf = sdf.drop(target, axis=1)    
    sdf = sdf.apply(lambda row: row.fillna(row.median()), axis=1)
    sdf_num = (sdf.values)[:, :ts_length]    
    if znormal:        
        sdf_num = znorm(sdf_num)    
    
    if split_flag==1:
        x_train, x_test, y_train, y_test = train_test_split(sdf_num, y_full, test_size=0.25, random_state=42)
    elif split_flag == 2:
        y_test = sdf_test[target].values.astype(int)
        sdf_test = sdf_test.drop(target, axis=1)    
        sdf_test = sdf_test.apply(lambda row: row.fillna(row.median()), axis=1)
        sdf_test_num = (sdf_test.values)[:, :ts_length]    
        if znormal:        
            sdf_test_num = znorm(sdf_test_num)
        x_train = sdf_num
        y_train = y_full
        x_test = sdf_test_num
        y_test = y_test        
    else:
        x_train = sdf_num
        y_train = y_full
        x_test = x_train.copy()
        y_test = y_train.copy()

    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)

    nb_classes = len(np.unique(y_train))
    if nb_classes > 1:
        y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
        y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    y_train = y_train.reshape(-1, 1).astype(int)
    y_test = y_test.reshape(-1, 1).astype(int)

    x_train = x_train[:, np.newaxis, :]
    x_test = x_test[:, np.newaxis, :]

    datasets_dict = {}
    datasets_dict[ds_name] = (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())
    return datasets_dict    
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ convert2cnn ] -------------------------------- #
def convert2cnn(sdf, target, ts_length, ds_name, sdf_test=None, split_flag=1, znormal=False):
    """
    Convert ECG/RR dataframe to CNN dataframe

    Args:
        sdf: standard dataframe with train data
        target: target name in sdf
        ts_length: time series length
        ds_name: name of dataset
        split_flag: 0 - no splitting, 1 - split to train and test, 2 - read sdf_test
        znormal: Z-normalization
        sdf_test: standard dataframe with test data        

    Returns:
        ts_X: DNN dataframe with X
        ts_y: ndarray with y
    """
    y_full = sdf[target].values.astype(int)
    sdf = sdf.drop(target, axis=1)    
    sdf = sdf.apply(lambda row: row.fillna(row.median()), axis=1)
    sdf_num = (sdf.values)[:, :ts_length]    
    if znormal:        
        sdf_num = znorm(sdf_num)
    
    if split_flag == 1:
        x_train, x_test, y_train, y_test = train_test_split(sdf_num, y_full, test_size=0.25, random_state=42)
    elif split_flag == 2:
        y_test = sdf_test[target].values.astype(int)
        sdf_test = sdf_test.drop(target, axis=1)    
        sdf_test = sdf_test.apply(lambda row: row.fillna(row.median()), axis=1)
        sdf_test_num = (sdf_test.values)[:, :ts_length]    
        if znormal:        
            sdf_test_num = znorm(sdf_test_num)
        x_train = sdf_num
        y_train = y_full
        x_test = sdf_test_num
        y_test = y_test
    else:
        x_train = sdf_num
        y_train = y_full
        x_test = x_train.copy()
        y_test = y_train.copy()

    datasets_dict = {}
    datasets_dict[ds_name] = (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())
    return datasets_dict    
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ convert2sktime ] -------------------------------- #
def convert2sktime(sdf, target, ts_length, znormal=False):
    """
    Convert ECG/RR dataframe to SKTIME dataframe (1 dimension: dim_0)

    Args:
        sdf: standard dataframe
        target: target name in sdf
        ts_length: time series length
        znormal: Z-normalization

    Returns:
        ts_X: sktime dataframe with X
        ts_y: ndarray with y
    """
    ts_y = sdf[target].values.astype(int).astype(str)
    sdf = sdf.drop(target, axis=1)    
    sdf = sdf.apply(lambda row: row.fillna(row.median()), axis=1)
    sdf_num = (sdf.values)[:, :ts_length]    
    if znormal:        
        sdf_num = znorm(sdf_num)    
   
    ts_list = list()
    for i in range(sdf_num.shape[0]):
        ts_list.append(pd.Series(sdf_num[i, :].astype(float)))
    
    ts_X = pd.DataFrame()
    ts_X['dim_0'] = ts_list   

    return ts_X, ts_y
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ read_dataset_lstmfcn_split2val_reduced ] -------------------------------- #
def read_dataset_lstmfcn_split2val_reduced(ds_name, path_to_data, start, stop):
    """
    Read train dataset from  local repository to lstmfcn dicts, split it to train and val(test) sets,
        reduced time length

    Args:
        ds_name: dataset name
        path_to_data: path to dataset
        start: start column
        stop: stop column

    Returns:
        x_train: x_train
        y_train: y_train
        x_test: x_test
        y_test: y_test
        is_timeseries: True

    """      
    total_path = path_to_data + ds_name
    train_path = os.path.join(total_path, ds_name) + '_TRAIN.ts'    

    X_full, y_full = load_from_tsfile_to_dataframe(train_path)

    X_full = get_all_instances_sktime_format(X_full)
    y_full = np.array(y_full).astype(int)
    
    x_train, x_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.25, random_state=42)

    x_train = x_train[:, start:stop]
    x_test = x_test[:, start:stop]
      
    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)

    nb_classes = len(np.unique(y_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    y_train = y_train.reshape(-1, 1).astype(int)
    y_test = y_test.reshape(-1, 1).astype(int)

    x_train = x_train[:, np.newaxis, :]
    x_test = x_test[:, np.newaxis, :]

    #x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    #x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))    
   
    return x_train, y_train, x_test, y_test, True
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ read_dataset_lstmfcn_reduced ] -------------------------------- #
def read_dataset_lstmfcn_reduced(ds_name, path_to_data, start, stop):
    """
    Read dataset from  local repository to lstmfcn dicts, reduced time length

    Args:
        ds_name: dataset name
        path_to_data: path to dataset
        start: start column
        stop: stop column

    Returns:
        x_train: x_train
        y_train: y_train
        x_test: x_test
        y_test: y_test
        is_timeseries: True

    """      
    total_path = path_to_data + ds_name
    train_path = os.path.join(total_path, ds_name) + '_TRAIN.ts'
    test_path = os.path.join(total_path, ds_name) + '_TEST.ts'

    X_train, y_train = load_from_tsfile_to_dataframe(train_path)
    X_test, y_test = load_from_tsfile_to_dataframe(test_path)
      
    x_train = get_all_instances_sktime_format(X_train)
    x_test = get_all_instances_sktime_format(X_test)

    x_train = x_train[:, start:stop]
    x_test = x_test[:, start:stop]

    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)

    nb_classes = len(np.unique(y_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    y_train = y_train.reshape(-1, 1).astype(int)
    y_test = y_test.reshape(-1, 1).astype(int)

    x_train = x_train[:, np.newaxis, :]
    x_test = x_test[:, np.newaxis, :]

    #x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    #x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))    
    
    return x_train, y_train, x_test, y_test, True
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ read_dataset_lstmfcn_split2val ] -------------------------------- #
def read_dataset_lstmfcn_split2val(ds_name, path_to_data):
    """
    Read train dataset from  local repository to lstmfcn dicts, split it to train and val(test) sets

    Args:
        ds_name: dataset name
        path_to_data: path to dataset

    Returns:
        x_train: x_train
        y_train: y_train
        x_test: x_test
        y_test: y_test
        is_timeseries: True

    """      
    total_path = path_to_data + ds_name
    train_path = os.path.join(total_path, ds_name) + '_TRAIN.ts'    

    X_full, y_full = load_from_tsfile_to_dataframe(train_path)

    X_full = get_all_instances_sktime_format(X_full)
    y_full = np.array(y_full).astype(int)
    
    x_train, x_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.25, random_state=42)
      
    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)

    nb_classes = len(np.unique(y_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    y_train = y_train.reshape(-1, 1).astype(int)
    y_test = y_test.reshape(-1, 1).astype(int)

    x_train = x_train[:, np.newaxis, :]
    x_test = x_test[:, np.newaxis, :]

    #x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    #x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))    
   
    return x_train, y_train, x_test, y_test, True
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ read_dataset_lstmfcn ] -------------------------------- #
def read_dataset_lstmfcn(ds_name, path_to_data):
    """
    Read dataset from  local repository to lstmfcn dicts

    Args:
        ds_name: dataset name
        path_to_data: path to dataset

    Returns:
        x_train: x_train
        y_train: y_train
        x_test: x_test
        y_test: y_test
        is_timeseries: True

    """      
    total_path = path_to_data + ds_name
    train_path = os.path.join(total_path, ds_name) + '_TRAIN.ts'
    test_path = os.path.join(total_path, ds_name) + '_TEST.ts'

    X_train, y_train = load_from_tsfile_to_dataframe(train_path)
    X_test, y_test = load_from_tsfile_to_dataframe(test_path)
      
    x_train = get_all_instances_sktime_format(X_train)
    x_test = get_all_instances_sktime_format(X_test)

    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)

    nb_classes = len(np.unique(y_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    y_train = y_train.reshape(-1, 1).astype(int)
    y_test = y_test.reshape(-1, 1).astype(int)

    x_train = x_train[:, np.newaxis, :]
    x_test = x_test[:, np.newaxis, :]

    #x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    #x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))    
    
    return x_train, y_train, x_test, y_test, True
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ class_activation_map ] -------------------------------- #
def class_activation_map(datasets_dict, dataset_name, output_directory):
    """
    Class Activation Map for Time Series Classification, save as PNG for each class
    Supports: resnet, inception

    Args:
        datasets_dict: dict with data in dl4tsc format
        dataset_name: dataset name
        output_directory: output directory with HDF files

    Returns:
        None (save PNG-file with CAM)

    """  
    max_length = 2000

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    y_test = datasets_dict[dataset_name][3]

    # transform to binary labels
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train_binary = enc.transform(y_train.reshape(-1, 1)).toarray()

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    path_to_best_model = output_directory + 'best_model.hdf5'
    model = keras.models.load_model(path_to_best_model)

    # filters
    w_k_c = model.layers[-1].get_weights()[0]  # weights for each filter k for each class c

    # the same input
    new_input_layer = model.inputs
    # output is both the original as well as the before last layer
    new_output_layer = [model.layers[-3].output, model.layers[-1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    classes = np.unique(y_train)

    for c in classes:
        plt.figure()
        count = 0
        c_x_train = x_train[np.where(y_train == c)]
        for ts in c_x_train:
            ts = ts.reshape(1, -1, 1)
            [conv_out, predicted] = new_feed_forward([ts])
            pred_label = np.argmax(predicted)
            orig_label = np.argmax(enc.transform([[c]]))
            if pred_label == orig_label:
                cas = np.zeros(dtype=np.float, shape=(conv_out.shape[1]))
                for k, w in enumerate(w_k_c[:, orig_label]):
                    cas += w * conv_out[0, :, k]

                minimum = np.min(cas)

                cas = cas - minimum

                cas = cas / max(cas)
                cas = cas * 100

                x = np.linspace(0, ts.shape[1] - 1, max_length, endpoint=True)
                # linear interpolation to smooth
                f = interp1d(range(ts.shape[1]), ts[0, :, 0])
                y = f(x)
                # if (y < -2.2).any():
                #     continue
                f = interp1d(range(ts.shape[1]), cas)
                cas = f(x).astype(int)
                plt.scatter(x=x, y=y, c=cas, cmap='jet', marker='.', s=2, vmin=0, vmax=100, linewidths=0.0)
                #if dataset_name == 'Gun_Point':
                #    if c == 1:
                #        plt.yticks([-1.0, 0.0, 1.0, 2.0])
                #    else:
                #        plt.yticks([-2, -1.0, 0.0, 1.0, 2.0])
                count += 1

        cbar = plt.colorbar()
        # cbar.ax.set_yticklabels([100,75,50,25,0])
        plt.savefig(output_directory + 'cam-class-' + str(int(c)) + '.png', bbox_inches='tight', dpi=1080)        
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ get_accuracy ] -------------------------------- #
def get_accuracy(path_to_best_model, datasets_dict, dataset_name):
    """
    Z-normalization

    Args:
        

    Returns:
        

    """  
    model = keras.models.load_model(path_to_best_model)
    y_train = datasets_dict[dataset_name][1]    
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()    
    y_true = np.argmax(y_test, axis=1)

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    accuracy= accuracy_score(y_true, y_pred)

    classic_metrics = dict()
    false_positive_rate, recall, thresholds = roc_curve(y_true.astype(int), y_pred.astype(int))
    roc_auc = auc(false_positive_rate, recall)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        classic_metrics['sensitivity'] = tp / (tp + fn)
        classic_metrics['specificity'] = tn / (tn + fp)
        classic_metrics['accuracy'] = (tn + tp) / (tn + tp + fn + fp)       
        classic_metrics['AUC'] = roc_auc
    except:
        classic_metrics['sensitivity'] = -1
        classic_metrics['specificity'] = -1
        classic_metrics['accuracy'] = -1
        classic_metrics['AUC'] = -1
    
    return accuracy, classic_metrics
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ fit_nn_classifier ] -------------------------------- #
def create_nn_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'lstmfcn':
        from classifiers import lstmfcn
        num_cells = 8
        seq_length = input_shape[1]
        return lstmfcn.Classifier_LSTMFCN(output_directory, input_shape, nb_classes, seq_length, num_cells, verbose)
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)        
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)
# -------------------------------- [ ] -------------------------------- #        

# -------------------------------- [ fit_nn_classifier ] -------------------------------- #
def fit_nn_classifier(classifier_name, datasets_dict, dataset_name, output_directory):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_nn_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)
    return classifier
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ read_train_dataset_dl4tsc_split2val_reduced ] -------------------------------- #
def read_train_dataset_dl4tsc_split2val_reduced(ds_name, path_to_data, start, stop):
    """
    Read train dataset from  local repository to dl4tsc dict and split it to train and val(test) sets,  
        reduced size of data

    Args:
        ds_name: dataset name
        path_to_data: path to dataset
        start: start column
        stop: stop column

    Returns:
        datasets_dict: dict with data
    """      
    total_path = path_to_data + ds_name
    train_path = os.path.join(total_path, ds_name) + '_TRAIN.ts'    
    X_full, y_full = load_from_tsfile_to_dataframe(train_path)

    X_full = get_all_instances_sktime_format(X_full)
    y_full = np.array(y_full).astype(int)
    
    x_train, x_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.25, random_state=42)

    x_train = x_train[:, start:stop]
    x_test = x_test[:, start:stop]
   
    datasets_dict = {}
    datasets_dict[ds_name] = (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())
    return datasets_dict
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ read_train_dataset_dl4tsc_split2val ] -------------------------------- #
def read_train_dataset_dl4tsc_split2val(ds_name, path_to_data):
    """
    Read train dataset from  local repository to dl4tsc dict and split it to train and val(test) sets

    Args:
        ds_name: dataset name
        path_to_data: path to dataset

    Returns:
        datasets_dict: dict with data
    """      
    total_path = path_to_data + ds_name
    train_path = os.path.join(total_path, ds_name) + '_TRAIN.ts'    
    X_full, y_full = load_from_tsfile_to_dataframe(train_path)

    X_full = get_all_instances_sktime_format(X_full)
    y_full = np.array(y_full).astype(int)
    
    x_train, x_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.25, random_state=42)
   
    datasets_dict = {}
    datasets_dict[ds_name] = (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())
    return datasets_dict
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ read_dataset_dl4tsc_reduced ] -------------------------------- #
def read_dataset_dl4tsc_reduced(ds_name, path_to_data, start, stop):
    """
    Read dataset from  local repository to dl4tsc dict, reduced size of data

    Args:
        ds_name: dataset name
        path_to_data: path to dataset
        start: start column
        stop: stop column

    Returns:
        datasets_dict: dict with data

    """      
    total_path = path_to_data + ds_name
    train_path = os.path.join(total_path, ds_name) + '_TRAIN.ts'
    test_path = os.path.join(total_path, ds_name) + '_TEST.ts'

    X_train, y_train = load_from_tsfile_to_dataframe(train_path)
    X_test, y_test = load_from_tsfile_to_dataframe(test_path)
      
    x_train = get_all_instances_sktime_format(X_train)
    x_test = get_all_instances_sktime_format(X_test)

    x_train = x_train[:, start:stop]
    x_test = x_test[:, start:stop]

    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)
    
    datasets_dict = {}
    datasets_dict[ds_name] = (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())
    return datasets_dict
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ read_dataset_dl4tsc ] -------------------------------- #
def read_dataset_dl4tsc(ds_name, path_to_data):
    """
    Read dataset from  local repository to dl4tsc dict

    Args:
        ds_name: dataset name
        path_to_data: path to dataset

    Returns:
        datasets_dict: dict with data

    """      
    total_path = path_to_data + ds_name
    train_path = os.path.join(total_path, ds_name) + '_TRAIN.ts'
    test_path = os.path.join(total_path, ds_name) + '_TEST.ts'

    X_train, y_train = load_from_tsfile_to_dataframe(train_path)
    X_test, y_test = load_from_tsfile_to_dataframe(test_path)
      
    x_train = get_all_instances_sktime_format(X_train)
    x_test = get_all_instances_sktime_format(X_test)

    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)
    
    datasets_dict = {}
    datasets_dict[ds_name] = (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())
    return datasets_dict
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ znorm ] -------------------------------- #
def znorm(x):
    """
    Z-normalization

    Args:
        x: ndarray with X data

    Returns:
        x: ndarray with X data after Z-normalization

    """  
    xstd = x.std(axis=1, keepdims=True)
    xstd[xstd == 0] = 1.0
    xmean = x.mean(axis=1, keepdims=True)        
    x = (x - xmean) / xstd
    return x
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ get_instance_sktime_format ] -------------------------------- #
def get_all_instances_sktime_format(tsdf, dim=0):
    """
    Get all time series instanses from sktime dataframe

    Args:
        tsdf: sktime dataframe
        inst_number: ordinal number of instance in dataframe
        dim: number of dataframe dimension (0 for univariate)

    Returns:
        ts: 2D numpy array with time series

    """  
    ts = list(tsdf['dim_' + str(dim)])
    return np.array(ts)
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ get_instance_sktime_format ] -------------------------------- #
def get_instance_sktime_format(tsdf, inst_number, dim=0):
    """
    Get time series instanse from sktime dataframe

    Args:
        tsdf: sktime dataframe
        inst_number: ordinal number of instance in dataframe
        dim: number of dataframe dimension (0 for univariate)

    Returns:
        ts: numpy array with time series

    """  
    ts = list(tsdf['dim_' + str(dim)][inst_number])
    return np.array(ts)
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ set_instance_sktime_format ] -------------------------------- #
def set_instance_sktime_format(tsdf, data, inst_number, dim=0):
    """
    Set new time series instanse to sktime dataframe

    Args:
        tsdf: sktime dataframe
        data: numpy array with new data (time series)
        inst_number: ordinal number of instance in dataframe
        dim: number of dataframe dimension (0 for univariate)

    Returns:
        tsdf: numpy array with time series

    """  
    temp_list = list(tsdf['dim_' + str(dim)])
    temp_list[inst_number] = pd.Series(data.astype(float))
    tsdf['dim_' + str(dim)] = temp_list
    return tsdf
# -------------------------------- [ ] -------------------------------- #