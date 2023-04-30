import os
import numpy as np
import pandas as pd
import urllib
import zipfile
import io
from sklearn import preprocessing

from DShap import DShap

def corrupt_label(y_train, noise_rate):
  """Corrupts training labels.
  Args:
    y_train: training labels
    noise_rate: input noise ratio
  Returns:
    corrupted_y_train: corrupted training labels
    noise_idx: corrupted index
  """

  y_set = list(set(y_train))

  # Sets noise_idx
  temp_idx = np.random.permutation(len(y_train))
  noise_idx = temp_idx[:int(len(y_train) * noise_rate)]

  # Corrupts label
  corrupted_y_train = y_train[:]

  for itt in noise_idx:
    temp_y_set = y_set[:]
    del temp_y_set[y_train[itt]]
    rand_idx = np.random.randint(len(y_set) - 1)
    corrupted_y_train[itt] = temp_y_set[rand_idx]

  return corrupted_y_train, noise_idx

def load_tabular_data(data_name="adult",dict_no=None, noise_rate=0):
    "Modified from https://github.com/google-research/google-research/blob/f027bdd54849b52d77194abdb7343a6d52a8e42c/dvrl/data_loading.py"
    """Loads Adult Income and Blog Feedback datasets.
      This module loads the two tabular datasets and saves train.csv, valid.csv and
      test.csv files under data_files directory.
      UCI Adult data link: https://archive.ics.uci.edu/ml/datasets/Adult
      UCI Blog data link: https://archive.ics.uci.edu/ml/datasets/BlogFeedback
      If noise_rate > 0.0, adds noise on the datasets.
      Then, saves train.csv, valid.csv, test.csv on './data_files/' directory
      Args:
        data_name: 'adult' or 'blog'
        dict_no: training and validation set numbers
        noise_rate: label corruption ratio
      Returns:
        noise_idx: indices of noisy samples
      """

    # Loads datasets from links
    uci_base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'

    # Adult Income dataset
    if data_name == 'adult':

        train_url = uci_base_url + 'adult/adult.data'
        test_url = uci_base_url + 'adult/adult.test'

        data_train = pd.read_csv(train_url, header=None)
        data_test = pd.read_csv(test_url, skiprows=1, header=None)

        df = pd.concat((data_train, data_test), axis=0)

        # Column names
        df.columns = ['Age', 'WorkClass', 'fnlwgt', 'Education', 'EducationNum',
                      'MaritalStatus', 'Occupation', 'Relationship', 'Race',
                      'Gender', 'CapitalGain', 'CapitalLoss', 'HoursPerWeek',
                      'NativeCountry', 'Income']

        # Creates binary labels
        df['Income'] = df['Income'].map({' <=50K': 0, ' >50K': 1,
                                         ' <=50K.': 0, ' >50K.': 1})

        # Changes string to float
        df.Age = df.Age.astype(float)
        df.fnlwgt = df.fnlwgt.astype(float)
        df.EducationNum = df.EducationNum.astype(float)
        df.EducationNum = df.EducationNum.astype(float)
        df.CapitalGain = df.CapitalGain.astype(float)
        df.CapitalLoss = df.CapitalLoss.astype(float)

        # One-hot encoding
        df = pd.get_dummies(df, columns=['WorkClass', 'Education', 'MaritalStatus',
                                         'Occupation', 'Relationship',
                                         'Race', 'Gender', 'NativeCountry'])

        # Sets label name as Y
        df = df.rename(columns={'Income': 'Y'})
        df['Y'] = df['Y'].astype(int)

        # Resets index
        df = df.reset_index()
        df = df.drop(columns=['index'])

    # Blog Feedback dataset
    elif data_name == 'blog':

        resp = urllib.request.urlopen(uci_base_url + '00304/BlogFeedback.zip')
        zip_file = zipfile.ZipFile(io.BytesIO(resp.read()))

        # Loads train dataset
        train_file_name = 'blogData_train.csv'
        data_train = pd.read_csv(zip_file.open(train_file_name), header=None)

        # Loads test dataset
        data_test = []
        for i in range(29):
            if i < 9:
                file_name = 'blogData_test-2012.02.0' + str(i + 1) + '.00_00.csv'
            else:
                file_name = 'blogData_test-2012.02.' + str(i + 1) + '.00_00.csv'

            temp_data = pd.read_csv(zip_file.open(file_name), header=None)

            if i == 0:
                data_test = temp_data
            else:
                data_test = pd.concat((data_test, temp_data), axis=0)

        for i in range(31):
            if i < 9:
                file_name = 'blogData_test-2012.03.0' + str(i + 1) + '.00_00.csv'
            elif i < 25:
                file_name = 'blogData_test-2012.03.' + str(i + 1) + '.00_00.csv'
            else:
                file_name = 'blogData_test-2012.03.' + str(i + 1) + '.01_00.csv'

            temp_data = pd.read_csv(zip_file.open(file_name), header=None)

            data_test = pd.concat((data_test, temp_data), axis=0)

        df = pd.concat((data_train, data_test), axis=0)

        # Removes rows with missing data
        df = df.dropna()

        # Sets label and named as Y
        df.columns = df.columns.astype(str)

        df['280'] = 1 * (df['280'] > 0)
        df = df.rename(columns={'280': 'Y'})
        df['Y'] = df['Y'].astype(int)

        # Resets index
        df = df.reset_index()
        df = df.drop(columns=['index'])

    # Splits train, valid and test sets
    train_idx = range(len(data_train))
    train = df.loc[train_idx]

    test_idx = range(len(data_train), len(df))
    test = df.loc[test_idx]

    train_idx_final = np.random.permutation(len(train))[:dict_no['train']]

    temp_idx = np.random.permutation(len(test))
    test_idx_final = temp_idx + len(data_train)

    train = train.loc[train_idx_final]
    test = test.loc[test_idx_final]

    # Adds noise on labels
    y_train = np.asarray(train['Y'])
    y_train, noise_idx = corrupt_label(y_train, noise_rate)
    train['Y'] = y_train

    # Saves data
    if not os.path.exists('data_files'):
        os.makedirs('data_files')

    train.to_csv('./data_files/train.csv', index=False)
    test.to_csv('./data_files/test.csv', index=False)

    # Returns indices of noisy samples
    return noise_idx

def preprocess_data(normalization,
                    train_file_name, test_file_name):
  "Modified from https://github.com/google-research/google-research/blob/f027bdd54849b52d77194abdb7343a6d52a8e42c/dvrl/data_loading.py"
  """Loads datasets, divides features and labels, and normalizes features.
  Args:
    normalization: 'minmax' or 'standard'
    train_file_name: file name of training set
    test_file_name: file name of testing set
  Returns:
    x_train: training features
    y_train: training labels
    x_valid: validation features
    y_valid: validation labels
    x_test: testing features
    y_test: testing labels
    col_names: column names
  """

  # Loads datasets
  train = pd.read_csv('./data_files/'+train_file_name)
  test = pd.read_csv('./data_files/'+test_file_name)

  # Extracts label
  y_train = np.asarray(train['Y'])
  y_test = np.asarray(test['Y'])

  # Drops label
  train = train.drop(columns=['Y'])
  test = test.drop(columns=['Y'])

  # Column names
  col_names = train.columns.values.astype(str)

  # Concatenates train, valid, test for normalization
  df = pd.concat((train, test), axis=0)

  # Normalization
  if normalization == 'minmax':
    scaler = preprocessing.MinMaxScaler()
  elif normalization == 'standard':
    scaler = preprocessing.StandardScaler()

  scaler.fit(df)
  df = scaler.transform(df)

  # Divides df into train / valid / test sets
  train_no = len(train)
  test_no = len(test)

  x_train = df[range(train_no), :]
  x_test = df[range(train_no, train_no+test_no), :]

  return x_train, y_train,  x_test, y_test, col_names

def get_synthetic_dataset():
    from shap_utils import return_model, label_generator

    problem, model = 'classification', 'logistic'
    hidden_units = []  # Empty list in the case of logistic regression.
    train_size = 1000

    d, difficulty = 50, 1
    num_classes = 2
    tol = 0.03
    target_accuracy = 0.7
    important_dims = 5
    clf = return_model(model, solver='liblinear', hidden_units=tuple(hidden_units))
    _param = 1.0
    for _ in range(100):
        X_raw = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d),
                                              size=train_size + 5000)
        _, y_raw, _, _ = label_generator(
            problem, X_raw, param=_param, difficulty=difficulty, important=important_dims)
        clf.fit(X_raw[:train_size], y_raw[:train_size])
        test_acc = clf.score(X_raw[train_size:], y_raw[train_size:])
        if test_acc > target_accuracy:
            break
        _param *= 1.1
    print('Performance using the whole training set = {0:.2f}'.format(test_acc))

    x_train, y_train = X_raw[:train_size], y_raw[:train_size]
    x_test, y_test = X_raw[train_size:], y_raw[train_size:]
    return x_train, y_train, x_test, y_test


def save_data(results_path,dataset_name,tol,x_train,y_train,x_test,y_test,dshap,num, tag=''):
    np.save(os.path.join(results_path, 'run_{}_{}_{}_x_train{}.npy'.format(num,dataset_name, tol,tag)), x_train)
    np.save(os.path.join(results_path, 'run_{}_{}_{}_y_train{}.npy'.format(num,dataset_name, tol,tag)), y_train)
    np.save(os.path.join(results_path, 'run_{}_{}_{}_x_test{}.npy'.format(num,dataset_name, tol,tag)), x_test)
    np.save(os.path.join(results_path, 'run_{}_{}_{}_y_test{}.npy'.format(num,dataset_name, tol,tag)), y_test)

    np.save(os.path.join(results_path, 'run_{}_{}_{}_vals_tmc{}.npy'.format(num,dataset_name, tol,tag)), dshap.vals_tmc)
    np.save(os.path.join(results_path, 'run_{}_{}_{}_vals_our{}.npy'.format(num,dataset_name, tol,tag)), dshap.vals_our)
    np.save(os.path.join(results_path, 'run_{}_{}_{}_vals_ourt{}.npy'.format(num, dataset_name, tol,tag)), dshap.vals_ourt)
    np.save(os.path.join(results_path, 'run_{}_{}_{}_vals_loo{}.npy'.format(num,dataset_name, tol,tag)), dshap.vals_loo)

    np.save(os.path.join(results_path, 'run_{}_{}_{}_time_tmc{}.npy'.format(num, dataset_name, tol,tag)), dshap.tmc_time)
    np.save(os.path.join(results_path, 'run_{}_{}_{}_time_our{}.npy'.format(num, dataset_name, tol,tag)), dshap.our_time)
    np.save(os.path.join(results_path, 'run_{}_{}_{}_time_ourt{}.npy'.format(num, dataset_name, tol,tag)), dshap.ourt_time)

def load_data(results_path,dataset_name,tol,model,directory,num_test,num, tag=''):
    x_train = np.load(os.path.join(results_path, 'run_{}_{}_{}_x_train{}.npy'.format(num,dataset_name, tol,tag)))
    y_train = np.load(os.path.join(results_path, 'run_{}_{}_{}_y_train{}.npy'.format(num,dataset_name, tol,tag)))
    x_test = np.load(os.path.join(results_path, 'run_{}_{}_{}_x_test{}.npy'.format(num,dataset_name, tol,tag)))
    y_test = np.load(os.path.join(results_path, 'run_{}_{}_{}_y_test{}.npy'.format(num,dataset_name, tol,tag)))

    dshap = DShap(x_train, y_train, x_test, y_test, num_test,
                  sources=None,
                  sample_weight=None,
                  model_family=model,
                  metric='accuracy',
                  overwrite=True,
                  directory=directory, seed=0)

    dshap.vals_tmc = np.load(os.path.join(results_path, 'run_{}_{}_{}_vals_tmc{}.npy'.format(num,dataset_name, tol,tag)))
    dshap.vals_our = np.load(os.path.join(results_path, 'run_{}_{}_{}_vals_our{}.npy'.format(num,dataset_name, tol,tag)))
    dshap.vals_ourt = np.load(os.path.join(results_path, 'run_{}_{}_{}_vals_ourt{}.npy'.format(num, dataset_name, tol,tag)))
    dshap.vals_loo = np.load(os.path.join(results_path, 'run_{}_{}_{}_vals_loo{}.npy'.format(num,dataset_name, tol,tag)))

    dshap.tmc_time = np.load(os.path.join(results_path, 'run_{}_{}_{}_time_tmc{}.npy'.format(num, dataset_name, tol,tag)))
    dshap.our_time = np.load(os.path.join(results_path, 'run_{}_{}_{}_time_our{}.npy'.format(num, dataset_name, tol,tag)))
    dshap.ourt_time = np.load(os.path.join(results_path, 'run_{}_{}_{}_time_ourt{}.npy'.format(num, dataset_name, tol,tag)))
    return x_train, y_train, x_test, y_test, dshap