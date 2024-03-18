import pandas as pd
import os
import numpy as np
import re
import sys
from collections import Counter
from scipy.special import expit
from itertools import compress
import pickle

class FeatureExtractor(object):

    def __init__(self):
        self.idf_vec = None
        self.mean_vec = None
        self.events = None
        self.term_weighting = None
        self.normalization = None
        self.oov = None

    def fit_transform(self, X_seq, term_weighting=None, normalization=None, oov=False, min_count=1):
        self.term_weighting = term_weighting
        self.normalization = normalization
        self.oov = oov

        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        self.events = X_df.columns
        X = X_df.values
        if self.oov:
            oov_vec = np.zeros(X.shape[0])
            if min_count > 1:
                idx = np.sum(X > 0, axis=0) >= min_count
                oov_vec = np.sum(X[:, ~idx] > 0, axis=1)
                X = X[:, idx]
                self.events = np.array(X_df.columns)[idx].tolist()
            X = np.hstack([X, oov_vec.reshape(X.shape[0], 1)])

        num_instance, num_event = X.shape
        if self.term_weighting == 'tf-idf':
            df_vec = np.sum(X > 0, axis=0)
            self.idf_vec = np.log(num_instance / (df_vec + 1e-8))
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))
            X = idf_matrix
        if self.normalization == 'zero-mean':
            mean_vec = X.mean(axis=0)
            self.mean_vec = mean_vec.reshape(1, num_event)
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == 'sigmoid':
            X[X != 0] = expit(X[X != 0])
        X_new = X

        print('log_data Train shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1]))
        return X_new

    def transform(self, X_seq):
        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        empty_events = set(self.events) - set(X_df.columns)
        for event in empty_events:
            X_df[event] = [0] * len(X_df)
        X = X_df[self.events].values
        if self.oov:
            oov_vec = np.sum(X_df[X_df.columns.difference(self.events)].values > 0, axis=1)
            X = np.hstack([X, oov_vec.reshape(X.shape[0], 1)])

        num_instance, num_event = X.shape
        if self.term_weighting == 'tf-idf':
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))
            X = idf_matrix
        if self.normalization == 'zero-mean':
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == 'sigmoid':
            X[X != 0] = expit(X[X != 0])
        X_new = X

        print('log_data Test shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1]))

        return X_new
import pandas as pd
import os
import numpy as np
import re
import sys
from sklearn.utils import shuffle
from collections import OrderedDict
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
def _split_data(x_data, y_data=None, train_ratio=0, split_type='uniform'):
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx]
        y_pos = y_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        train_pos = int(train_ratio * x_pos.shape[0])
        train_neg = int(train_ratio * x_neg.shape[0])
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
        y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
    elif split_type == 'sequential':
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]

    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    return (x_train, y_train), (x_test, y_test)
def load_HDFS(log_file, label_file=None, window='session', train_ratio=0.5, split_type='sequential',
    save_csv=False):


    if log_file.endswith('.csv'):
        assert window == 'session', "Only window=session is supported for HDFS dataset."
        struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
        data_dict = OrderedDict()
        for idx, row in struct_log.iterrows():
            blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row['EventId'])
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

        if label_file:

            label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
            label_data = label_data.set_index('BlockId')
            label_dict = label_data['Label'].to_dict()
            data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
            (x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values,
                data_df['Label'].values, train_ratio, split_type)

        if save_csv:
            data_df.to_csv('data_instances.csv', index=False)

        if label_file is None:
            if split_type == 'uniform':
                split_type = 'sequential'
                print('Warning: Only split_type=sequential is supported \
                if label_file=None.'.format(split_type))
            x_data = data_df['EventSequence'].values
            (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type)
            print('Total: {} instances, train: {} instances, test: {} instances'.format(
                  x_data.shape[0], x_train.shape[0], x_test.shape[0]))
            return (x_train, None), (x_test, None)
    else:
        raise NotImplementedError('load_HDFS() can support only csv and npz files!')

    no_train = x_train.shape[0]
    no_test = x_test.shape[0]
    no_total = no_train + no_test
    no_train_pos = sum(y_train)
    no_test_pos = sum(y_test)
    no_pos = no_train_pos + no_test_pos

    print('Overall: {} instances, {} anomaly, {} normal' \
          .format(no_total, no_pos, no_total - no_pos))
    print('from Train: {} instances, {} anomaly, {} normal' \
          .format(no_train, no_train_pos, no_train - no_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(no_test, no_test_pos, no_test - no_test_pos))

    return (x_train, y_train), (x_test, y_test)
import pandas as pd
struct_log='HDFS.log_structured.csv'
labels='anomaly_labels.csv'
if __name__=='__main__':
  (x_tr, y_train), (x_te, y_test) = load_HDFS(struct_log,label_file=labels,window='session',train_ratio=0.7,split_type='uniform')
feature_extractor =FeatureExtractor()
x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
x_test=feature_extractor.transform(x_te)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
with open("model.pkl", "wb") as f:
    pickle.dump((model, x_test, y_test), f)
from sklearn.svm import SVC
model_1=SVC()
model_1.fit(x_train,y_train)
with open("model_1.pkl", "wb") as f:
    pickle.dump((model_1, x_test, y_test), f)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
with open("model_3.pkl", "wb") as f:
    pickle.dump((lr, x_test, y_test), f)