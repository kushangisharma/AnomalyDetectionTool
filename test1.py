from flask import Flask, render_template, request
import pandas as pd
import re
import numpy as np
from sklearn.utils import shuffle
from collections import OrderedDict
import pickle 
from sklearn.metrics import accuracy_score,precision_score,f1_score
app = Flask(__name__)
with open("model.pkl", "rb") as f:
    model, X_test, Y_test = pickle.load(f)
with open("model_1.pkl", "rb") as f:
    model_1, X_test_1, Y_test_1 = pickle.load(f)
with open("model_3.pkl", "rb") as f:
    model_3, X_test_3, Y_test_3 = pickle.load(f)

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
        raise NotImplementedError('load_HDFS() only support csv and npz files!')

    no_train = x_train.shape[0]
    no_test = x_test.shape[0]
    no_total = no_train + no_test
    no_train_pos = sum(y_train)
    no_test_pos = sum(y_test)
    no_pos = no_train_pos + no_test_pos

    # print('Overall: {} instances, {} anomaly, {} normal' \
    #       .format(no_total, no_pos, no_total - no_pos))
    # print('from Train: {} instances, {} anomaly, {} normal' \
    #       .format(no_train, no_train_pos, no_train - no_train_pos))
    # print('Test: {} instances, {} anomaly, {} normal\n' \
    #       .format(no_test, no_test_pos, no_test - no_test_pos))

    return (x_train, y_train), (x_test, y_test)
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/process_files', methods=['POST'])
def process_files():
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if file1.filename == '' or file2.filename == '':
        return 'Please select two CSV files.'
    
    
    content1 = file1.read().decode('utf-8')
    content2 = file2.read().decode('utf-8')
    struct_log=file1.filename
    labels=file2.filename
    (x_train,y_train),(x_test,y_test)=load_HDFS(struct_log,label_file=labels,window='session',train_ratio=0.8,split_type='uniform')
    df4=pd.read_csv(struct_log)
    arr=df4['EventId'].unique()
    event_count=list(arr)
    length_of_event_count=len(event_count)
    event_count_matrix=[]
    for row in x_train:
        matrix_part=np.zeros(length_of_event_count)
        for j in row:
            find_ind=event_count.index(j)
            matrix_part[find_ind]=matrix_part[find_ind]+1
        event_count_matrix.append(matrix_part)
    print(event_count_matrix)
    df8=pd.DataFrame(event_count_matrix)
    df8.columns=event_count
    y_pred_1=model.predict(X_test)
    decision_tree_accuracy=accuracy_score(Y_test,y_pred_1)
    decision_tree_precision=precision_score(Y_test,y_pred_1)
    decision_tree_f1_score=f1_score(Y_test,y_pred_1)
    y_pred_2=model_1.predict(X_test_1)
    svm_accuracy=accuracy_score(Y_test_1,y_pred_2)
    svm_precision=precision_score(Y_test_1,y_pred_2)
    svm_f1_score=f1_score(Y_test_1,y_pred_2)
    y_pred_3=model_3.predict(X_test_3)
    lr_accuracy=accuracy_score(Y_test_3,y_pred_3)
    lr_precision=precision_score(Y_test_3,y_pred_3)
    lr_f1_score=f1_score(Y_test_3,y_pred_3)
    
    
    
    return render_template('result.html', event_count_matrix=df8.to_html(),
                           decision_tree_accuracy=decision_tree_accuracy,
                           decision_tree_precision=decision_tree_precision,
                           decision_tree_f1_score=decision_tree_f1_score,
                           svm_accuracy=svm_accuracy,
                           svm_precision=svm_precision,
                           svm_f1_score=svm_f1_score,
                           lr_accuracy=lr_accuracy,
                           lr_precision=lr_precision,
                           lr_f1_score=lr_f1_score)

