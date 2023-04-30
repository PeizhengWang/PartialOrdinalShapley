import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()

from DShap import DShap
from data.load_data import load_tabular_data, load_data, preprocess_data, get_synthetic_dataset, save_data, corrupt_label




def args_parser():
    # Default settings
    parser = argparse.ArgumentParser(description='Partial Ordinal Shapley')
    parser.add_argument('--dataset_name', type=str, default="cancer", help="which dataset['wine', 'cancer', 'adult_s']")
    parser.add_argument('--model_name', type=str, default="logistic", help='choose a learner')
    parser.add_argument('--results_path', type=str, default='./results', help='path for saving results')
    parser.add_argument('--recompute_shapley', type=bool, default=True, help='Compute from scratch or just draw')
    parser.add_argument('--tol', type=float, default=0.05, help='Truncated factor')
    parser.add_argument('--start_run', type=int, default=0, help='start run num')
    parser.add_argument('--num_run', type=int, default=1, help='num of experiment')
    parser.add_argument('--ratio', type=float, default=0.8, help='ratio')
    parser.add_argument('--noise', type=float, default=0, help='noise ratio')
    args = parser.parse_args()
    return args


def data_shapley(args):
    dataset_name = args.dataset_name
    model = args.model_name
    results_path = args.results_path
    recompute_shapley = args.recompute_shapley
    tol = args.tol
    start_run = args.start_run
    num_run = args.num_run
    ratio = args.ratio
    noise = args.noise

    if noise==0:
        tag=""
    else:
        tag="_noise_%s" %(noise)

    if ratio!=0.8:
        tag=tag+'ratio_'+str(ratio)

    if dataset_name == 'wine':
        num_test = 40
    elif dataset_name == 'cancer':
        num_test = 100
    else:
        num_test = 1000


    for num in range(start_run, start_run+num_run):
        print(num)
        if recompute_shapley:

            if dataset_name == 'wine':
                dataset = datasets.load_wine()  # chemical
                feature = np.array(dataset.data)
                label = np.array(dataset.target)

                label,_=corrupt_label(label,noise)

                x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.5)  # 划分测试集和训练集

            elif dataset_name == 'cancer':
                dataset = datasets.load_breast_cancer()  # medical
                feature = np.array(dataset.data)
                label = np.array(dataset.target)
                label,_ = corrupt_label(label, noise)
                x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.5)  # 划分测试集和训练集

            elif dataset_name == 'synthetic':
                x_train, y_train, x_test, y_test = get_synthetic_dataset()
                num_test = 1000
            elif dataset_name == 'adult':
                dict_no = dict()
                dict_no['train'] = 1000
                dict_no['valid'] = 400

                if noise == 0:
                    _ = load_tabular_data('adult', dict_no, 0)
                else:
                    _ = load_tabular_data('adult',dict_no, 0.2)
                x_train, y_train, x_test,  y_test, _ = preprocess_data('standard','train.csv', 'test.csv')
                # print(y_train)
                num_test = len(y_test)-dict_no['valid']

            elif dataset_name == 'adult_s':
                dict_no = dict()
                dict_no['train'] = 200
                dict_no['valid'] = 200

                if noise == 0:
                    _ = load_tabular_data('adult', dict_no, 0)
                else:
                    _ = load_tabular_data('adult', dict_no, 0.2)

                x_train, y_train, x_test,  y_test, _ = preprocess_data('standard','train.csv', 'test.csv')
                num_test = len(y_test)-dict_no['valid']

            else:
                print('Invalid dataset_name!')
                return

            dshap = DShap(x_train, y_train, x_test, y_test, num_test,
                          sources=None,
                          sample_weight=None,
                          model_family=model,
                          metric='accuracy',
                          overwrite=True,
                          directory='./temp', seed=None,
                          n_jobs=-1,
                          ratio=ratio)
            dshap.run(100, tol, g_run=False, loo_run=True, our_run=True,ourt_run=True)
            save_data(results_path, dataset_name, tol, x_train, y_train, x_test, y_test, dshap, num, tag)
        else:
            x_train,y_train,x_test,y_test,dshap=load_data(results_path,dataset_name,tol,model,'./temp',num_test,num,tag)


        perfs = dshap.performance_plots([dshap.vals_ourt, dshap.vals_our, dshap.vals_tmc],
                                        num_plot_markers=20,
                                        sources=dshap.sources)
        if recompute_shapley:

            plt.savefig('run_{}_{}_shapley{}.jpg'.format(dataset_name,num,tag))
            plt.clf()
            print('ctmc time:',dshap.ourt_time,'ctmc area(50):',np.sum(perfs[0][:int(len(perfs[0]) * 0.5)]),'ctmc area(100):',np.sum(perfs[0]))
            print('cmt time:', dshap.our_time,'cmt area(50):',np.sum(perfs[1][:int(len(perfs[1]) * 0.5)]),'cmt area(100):',np.sum(perfs[1]))
            print('tmc time:', dshap.tmc_time, 'tmc area(50):', np.sum(perfs[2][:int(len(perfs[2]) * 0.5)]),'tmc area(100):', np.sum(perfs[2]))

            text1='ctmc time:'+str(dshap.ourt_time)+'ctmc area(50):'+str(np.sum(perfs[0][:int(len(perfs[0]) * 0.5)]))+'ctmc area(100):'+str(np.sum(perfs[0]))
            text2='cmt time:'+str(dshap.our_time)+'cmt area(50):'+str(np.sum(perfs[1][:int(len(perfs[1]) * 0.5)]))+'cmt area(100):'+str(np.sum(perfs[1]))
            text3='tmc time:'+str(dshap.tmc_time)+'tmc area(50):'+str(np.sum(perfs[2][:int(len(perfs[2]) * 0.5)]))+'tmc area(100):'+str(np.sum(perfs[2]))

            f = open("log_{}_{}{}.txt".format(dataset_name,tol,tag), "a")
            f.write(text1)
            f.write(text2)
            f.write(text3)
            f.close()

        else:

            plt.savefig('run_{}_{}_shapley_redraw{}.jpg'.format(dataset_name, num,tag))
            plt.clf()

            print('ctmc time:', dshap.ourt_time, 'ctmc area(50):', np.sum(perfs[0][:int(len(perfs[0]) * 0.5)]),
                  'ctmc area(100):', np.sum(perfs[0]))
            print('cmt time:', dshap.our_time, 'cmt area(50):', np.sum(perfs[1][:int(len(perfs[1]) * 0.5)]),
                  'cmt area(100):', np.sum(perfs[1]))
            print('tmc time:', dshap.tmc_time, 'tmc area(50):', np.sum(perfs[2][:int(len(perfs[2]) * 0.5)]),
                  'tmc area(100):', np.sum(perfs[2]))




if __name__ == '__main__':
    args = args_parser()
    # print(args)
    data_shapley(args)


