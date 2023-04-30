
#______________________________________PEP8____________________________________
#_______________________________________________________________________
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import tensorflow as tf
import sys
from shap_utils import *
from Shapley import ShapNN
from scipy.stats import spearmanr
import shutil
from sklearn.base import clone
import matplotlib.pyplot as plt
import warnings
import itertools
import inspect
import _pickle as pkl
from sklearn.metrics import f1_score, roc_auc_score
import math

import tensorflow
import time


class DShap(object):
    
    def __init__(self, X, y, X_test, y_test, num_test, sources=None, 
                 sample_weight=None, directory=None, problem='classification',
                 model_family='logistic', metric='accuracy', seed=None,
                 overwrite=False,
                 **kwargs):
        """
        Args:
            X: Data covariates
            y: Data labels
            X_test: Test+Held-out covariates
            y_test: Test+Held-out labels
            sources: An array or dictionary assiging each point to its group.
                If None, evey points gets its individual value.
            samples_weights: Weight of train samples in the loss function
                (for models where weighted training method is enabled.)
            num_test: Number of data points used for evaluation metric.
            directory: Directory to save results and figures.
            problem: "Classification" or "Regression"(Not implemented yet.)
            model_family: The model family used for learning algorithm
            metric: Evaluation metric
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting 
                same permutations.
            overwrite: Delete existing data and start computations from 
                scratch
            **kwargs: Arguments of the model
        """
            
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_random_seed(seed)
        self.problem = problem
        self.model_family = model_family
        self.metric = metric
        self.directory = directory
        self.hidden_units = kwargs.get('hidden_layer_sizes', [])
        self.ratio = kwargs.get('ratio', 1)
        count = 0
        for cs in set(y):
            count += int((y == cs).sum() * self.ratio)
        self.count = count
        if self.model_family is 'logistic':
            self.hidden_units = []
        if self.directory is not None:
            if overwrite and os.path.exists(directory):
                tf.gfile.DeleteRecursively(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)  
                os.makedirs(os.path.join(directory, 'weights'))
                os.makedirs(os.path.join(directory, 'plots'))
            self._initialize_instance(X, y, X_test, y_test, num_test, 
                                      sources, sample_weight)
        if len(set(self.y)) > 2:
            assert self.metric != 'f1', 'Invalid metric for multiclass!'
            assert self.metric != 'auc', 'Invalid metric for multiclass!'
        is_regression = (np.mean(self.y//1 == self.y) != 1)
        is_regression = is_regression or isinstance(self.y[0], np.float32)
        self.is_regression = is_regression or isinstance(self.y[0], np.float64)
        if self.is_regression:
            warnings.warn("Regression problem is no implemented.")
        self.model = return_model(self.model_family, **kwargs)
        self.random_score = self.init_score(self.metric)


        self.tmc_time=0.0
        self.our_time=0.0
        self.ourt_time = 0.0

    def _initialize_instance(self, X, y, X_test, y_test, num_test, 
                             sources=None, sample_weight=None):
        """Loads or creates sets of data."""      
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        data_dir = os.path.join(self.directory, 'data.pkl')
        if os.path.exists(data_dir):
            self._load_dataset(data_dir)
        else:
            self.X_heldout = X_test[:-num_test]
            self.y_heldout = y_test[:-num_test]
            self.X_test = X_test[-num_test:]
            self.y_test = y_test[-num_test:]
            self.X, self.y, self.sources = X, y, sources
            self.sample_weight = sample_weight
            data_dic = {'X': self.X, 'y': self.y, 'X_test': self.X_test,
                     'y_test': self.y_test, 'X_heldout': self.X_heldout,
                     'y_heldout':self.y_heldout, 'sources': self.sources}
            if sample_weight is not None:
                data_dic['sample_weight'] = sample_weight
                warnings.warn("Sample weight not implemented for G-Shapley")
            pkl.dump(data_dic, open(data_dir, 'wb'))        
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        self.vals_loo = None
        if os.path.exists(loo_dir):
            self.vals_loo = pkl.load(open(loo_dir, 'rb'))['loo']
        n_sources = len(self.X) if self.sources is None else len(self.sources)
        n_points = len(self.X)
        self.tmc_number, self.g_number , self.our_number, self.ourt_number = self._which_parallel(self.directory)
        self._create_results_placeholder(
            self.directory, self.tmc_number, self.g_number, self.our_number,self.ourt_number,
            n_points, n_sources, self.model_family)
        
    def _create_results_placeholder(self, directory, tmc_number, g_number, our_number,ourt_number,
                                   n_points, n_sources, model_family):
        tmc_dir = os.path.join(
            directory, 
            'mem_tmc_{}.pkl'.format(tmc_number.zfill(4))
        )
        g_dir = os.path.join(
            directory,
            'mem_g_{}.pkl'.format(g_number.zfill(4))
        )
        our_dir = os.path.join(
            directory,
            'mem_our_{}.pkl'.format(our_number.zfill(4))
        )
        ourt_dir = os.path.join(
            directory,
            'mem_our_{}.pkl'.format(ourt_number.zfill(4))
        )
        self.mem_tmc = np.zeros((0, n_points))
        self.mem_g = np.zeros((0, n_points))
        self.mem_our = np.zeros((0, n_points))
        self.mem_ourt = np.zeros((0, n_points))
        self.idxs_tmc = np.zeros((0, n_sources), int)
        self.idxs_g = np.zeros((0, n_sources), int)
        self.idxs_our = np.zeros((0, self.count), int)
        self.idxs_ourt = np.zeros((0, self.count), int)
        #0.5： 44 142 499
        #0.6:170
        #0.7:197
        #0.8： 227    799 for adult
        #0.9:255
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc},
                 open(tmc_dir, 'wb'))
        pkl.dump({'mem_our': self.mem_our, 'idxs_our': self.idxs_our},
                 open(our_dir, 'wb'))
        pkl.dump({'mem_ourt': self.mem_ourt, 'idxs_ourt': self.idxs_our},
                 open(ourt_dir, 'wb'))
        if model_family not in ['logistic', 'NN']:
            return
        pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g}, 
                 open(g_dir, 'wb'))
        
    def _load_dataset(self, data_dir):
        '''Load the different sets of data if already exists.'''
        data_dic = pkl.load(open(data_dir, 'rb'))
        self.X_heldout = data_dic['X_heldout']
        self.y_heldout = data_dic['y_heldout']
        self.X_test = data_dic['X_test']
        self.y_test = data_dic['y_test']
        self.X = data_dic['X'] 
        self.y = data_dic['y']
        self.sources = data_dic['sources']
        if 'sample_weight' in data_dic.keys():
            self.sample_weight = data_dic['sample_weight']
        else:
            self.sample_weight = None
        
    def _which_parallel(self, directory):
        '''Prevent conflict with parallel runs.'''
        previous_results = os.listdir(directory)
        tmc_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                     for name in previous_results if 'mem_tmc' in name]
        our_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                     for name in previous_results if 'mem_our' in name]
        ourt_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                     for name in previous_results if 'mem_ourt' in name]
        g_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                     for name in previous_results if 'mem_g' in name]        
        tmc_number = str(np.max(tmc_nmbrs) + 1) if len(tmc_nmbrs) else '0'
        our_number = str(np.max(our_nmbrs) + 1) if len(our_nmbrs) else '0'
        ourt_number = str(np.max(our_nmbrs) + 1) if len(ourt_nmbrs) else '0'
        g_number = str(np.max(g_nmbrs) + 1) if len(g_nmbrs) else '0' 
        return tmc_number, g_number, our_number, ourt_number
    
    def init_score(self, metric):
        """ Gives the value of an initial untrained model."""
        if metric == 'accuracy':
            hist = np.bincount(self.y_test).astype(float)/len(self.y_test)
            return np.max(hist)
        if metric == 'f1':
            rnd_f1s = []
            for _ in range(1000):
                rnd_y = np.random.permutation(self.y_test)
                rnd_f1s.append(f1_score(self.y_test, rnd_y))
            return np.mean(rnd_f1s)
        if metric == 'auc':
            return 0.5
        random_scores = []
        for _ in range(100):
            rnd_y = np.random.permutation(self.y)
            if self.sample_weight is None:
                self.model.fit(self.X, rnd_y)
            else:
                self.model.fit(self.X, rnd_y, 
                               sample_weight=self.sample_weight)
            random_scores.append(self.value(self.model, metric))
        return np.mean(random_scores)
        
    def value(self, model, metric=None, X=None, y=None):
        """Computes the values of the given model.
        Args:
            model: The model to be evaluated.
            metric: Valuation metric. If None the object's default
                metric is used.
            X: Covariates, valuation is performed on a data 
                different from test set.
            y: Labels, if valuation is performed on a data 
                different from test set.
            """
        if metric is None:
            metric = self.metric
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
        if inspect.isfunction(metric):
            return metric(model, X, y)
        if metric == 'accuracy':
            return model.score(X, y)
        if metric == 'f1':
            assert len(set(y)) == 2, 'Data has to be binary for f1 metric.'
            return f1_score(y, model.predict(X))
        if metric == 'auc':
            assert len(set(y)) == 2, 'Data has to be binary for auc metric.'
            return my_auc_score(model, X, y)
        if metric == 'xe':
            return my_xe_score(model, X, y)
        raise ValueError('Invalid metric!')
        
    def run(self, save_every, err, tolerance=0.01, g_run=True, loo_run=True,our_run=False,ourt_run=False):
        """Calculates data sources(points) values.
        
        Args:
            save_every: save marginal contrivbutions every n iterations.
            err: stopping criteria.
            tolerance: Truncation tolerance. If None, it's computed.
            g_run: If True, computes G-Shapley values.
            loo_run: If True, computes and saves leave-one-out scores.
        """
        if loo_run:
            try:
                len(self.vals_loo)
            except:
                self.vals_loo = self._calculate_loo_vals(sources=self.sources)
                self.save_results(overwrite=True)
        print('LOO values calculated!')
        tmc_run = True 
        g_run = g_run and self.model_family in ['logistic', 'NN']
        while tmc_run or g_run or our_run or ourt_run:
            # if not our_run:
            #     break
            if g_run:
                print(error(self.mem_g))
                if error(self.mem_g) < err:
                    g_run = False
                else:
                    self._g_shap(save_every, sources=self.sources)
                    self.vals_g = np.mean(self.mem_g, 0)
            if tmc_run:
                time_start = time.time()
                print(len(self.mem_tmc),error(self.mem_tmc))
                if error(self.mem_tmc) < err:
                    tmc_run = False
                else:
                    self._tmc_shap(
                        save_every, 
                        tolerance=tolerance, 
                        sources=self.sources
                    )
                    self.vals_tmc = np.mean(self.mem_tmc, 0)
                time_end = time.time()
                self.tmc_time+=time_end-time_start
            if our_run:
                time_start = time.time()
                print(len(self.mem_our), error(self.mem_our))
                if error(self.mem_our) < err:
                    our_run = False
                else:
                    self._our_shap(
                        save_every,
                        tolerance=tolerance,
                        sources=self.sources
                    )
                    self.vals_our = np.mean(self.mem_our, 0)
                time_end = time.time()
                self.our_time += time_end - time_start
            if ourt_run:
                time_start = time.time()
                print(len(self.mem_ourt), error(self.mem_ourt))
                if error(self.mem_ourt) < err:
                    ourt_run = False
                else:
                    self._ourt_shap(
                        save_every,
                        tolerance=tolerance,
                        sources=self.sources
                    )
                    self.vals_ourt = np.mean(self.mem_ourt, 0)
                time_end = time.time()
                self.ourt_time += time_end - time_start

            if self.directory is not None:
                self.save_results()
            print('tmc time:',self.tmc_time,'our time:',self.our_time,'ourt time:',self.ourt_time)
    def save_results(self, overwrite=False):
        """Saves results computed so far."""
        if self.directory is None:
            return
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        if not os.path.exists(loo_dir) or overwrite:
            pkl.dump({'loo': self.vals_loo}, open(loo_dir, 'wb'))
        tmc_dir = os.path.join(
            self.directory,
            'mem_tmc_{}.pkl'.format(self.tmc_number.zfill(4))
        )
        g_dir = os.path.join(
            self.directory, 
            'mem_g_{}.pkl'.format(self.g_number.zfill(4))
        )
        our_dir = os.path.join(
            self.directory,
            'mem_our_{}.pkl'.format(self.tmc_number.zfill(4))
        )
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc},
                 open(tmc_dir, 'wb'))
        pkl.dump({'mem_our': self.mem_our, 'idxs_our': self.idxs_our},
                 open(our_dir, 'wb'))
        pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g}, 
                 open(g_dir, 'wb'))  
        
    def _tmc_shap(self, iterations, tolerance=None, sources=None):
        """Runs TMC-Shapley algorithm.
        
        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance ratio.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        model = self.model
        try:
            self.mean_score
        except:
            self._tol_mean_score()
        if tolerance is None:
            tolerance = self.tolerance         
        marginals, idxs = [], []
        for iteration in range(iterations):
            if 10*(iteration+1)/iterations % 1 == 0:
                print('{} out of {} TMC_Shapley iterations.'.format(
                    iteration + 1, iterations))
            marginals, idxs = self.one_iteration(
                tolerance=tolerance, 
                sources=sources
            )
            # print(len(marginals))
            self.mem_tmc = np.concatenate([
                self.mem_tmc, 
                np.reshape(marginals, (1,-1))
            ])
            self.idxs_tmc = np.concatenate([
                self.idxs_tmc, 
                np.reshape(idxs, (1,-1))
            ])

    def _our_shap(self, iterations, tolerance=None, sources=None):
        """Runs TMC-Shapley algorithm.

        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance ratio.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        model = self.model
        try:
            self.mean_score
        except:
            self._tol_mean_score()
        if tolerance is None:
            tolerance = self.tolerance
        marginals, idxs = [], []
        for iteration in range(iterations):
            if 10 * (iteration + 1) / iterations % 1 == 0:
                print('{} out of {} our_Shapley iterations.'.format(
                    iteration + 1, iterations))
            marginals, idxs = self.our_one_iteration(
                tolerance=tolerance,
                sources=sources
            )
            # print(len(marginals))
            self.mem_our = np.concatenate([
                self.mem_our,
                np.reshape(marginals, (1, -1))
            ])
            self.idxs_our = np.concatenate([
                self.idxs_our,
                np.reshape(idxs, (1, -1))
            ])

    def _ourt_shap(self, iterations, tolerance=None, sources=None):
        """Runs TMC-Shapley algorithm.

        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance ratio.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        model = self.model
        try:
            self.mean_score
        except:
            self._tol_mean_score()
        if tolerance is None:
            tolerance = self.tolerance
        marginals, idxs = [], []
        for iteration in range(iterations):
            if 10 * (iteration + 1) / iterations % 1 == 0:
                print('{} out of {} ourt_Shapley iterations.'.format(
                    iteration + 1, iterations))
            marginals, idxs = self.ourt_one_iteration(
                tolerance=tolerance,
                sources=sources
            )
            # print(len(marginals))
            self.mem_ourt = np.concatenate([
                self.mem_ourt,
                np.reshape(marginals, (1, -1))
            ])
            self.idxs_ourt = np.concatenate([
                self.idxs_ourt,
                np.reshape(idxs, (1, -1))
            ])

    def _tol_mean_score(self):
        """Computes the average performance and its error using bagging."""
        scores = []
        self.restart_model()
        for _ in range(1):
            num = len(self.X)
            x_g = np.arange(num)
            mu = (num - 1) / 2
            sigma = (num - 1) / 6
            sample_weight_batch = np.exp(-1 * ((x_g - mu) ** 2) / (2 * (sigma ** 2))) / (
                        math.sqrt(2 * np.pi) * sigma) * num

            self.model.fit(self.X, self.y,
                          sample_weight=sample_weight_batch)
            for _ in range(100):
                bag_idxs = np.random.choice(len(self.y_test), len(self.y_test))
                scores.append(self.value(
                    self.model, 
                    metric=self.metric,
                    X=self.X_test[bag_idxs], 
                    y=self.y_test[bag_idxs]
                ))
        self.tol = np.std(scores)
        self.mean_score = np.mean(scores)
        
    def one_iteration(self, tolerance, sources=None):
        """Runs one iteration of TMC-Shapley algorithm."""
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        idxs = np.random.permutation(len(sources))
        marginal_contribs = np.zeros(len(self.X))
        X_batch = np.zeros((0,) + tuple(self.X.shape[1:]))
        y_batch = np.zeros(0, int)
        sample_weight_batch = np.zeros(0)
        truncation_counter = 0
        new_score = self.random_score
        # self.restart_model()
        for n, idx in enumerate(idxs):
            old_score = new_score
            X_batch = np.concatenate([X_batch, self.X[sources[idx]]])
            y_batch = np.concatenate([y_batch, self.y[sources[idx]]])
            if self.sample_weight is None:
                sample_weight_batch = None
            else:
                sample_weight_batch = np.concatenate([
                    sample_weight_batch, 
                    self.sample_weight[sources[idx]]
                ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if (self.is_regression 
                    or len(set(y_batch)) == len(set(self.y_test))): ##FIXIT
                    self.restart_model()

                    num=len(X_batch)
                    x_g=np.arange(num)
                    mu=(num-1)/2
                    sigma=(num-1)/6
                    sample_weight_batch=np.exp(-1*((x_g-mu)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)*num
                    self.model.fit(
                        X_batch,
                        y_batch,
                        sample_weight = sample_weight_batch
                    )
                    new_score = self.value(self.model, metric=self.metric)       
            marginal_contribs[sources[idx]] = (new_score - old_score)
            marginal_contribs[sources[idx]] /= len(sources[idx])
            # print(n, new_score)
            distance_to_full_score = np.abs(new_score - self.mean_score)
            if distance_to_full_score <= tolerance * self.mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    # print(n)
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs

    def our_one_iteration(self, tolerance, sources=None):
        """Runs one iteration of our-Shapley algorithm."""
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}


        idxs = np.random.permutation(len(sources))


        y=self.y[idxs]
        class_list=set(y)
        idxs_list=[]
        for cs in class_list:

            tmp=idxs[y==cs]
            # print(cs, int(len(tmp) / 2))
            idxs_list.append(tmp[:int(len(tmp)*self.ratio)])
        idxs= np.concatenate(idxs_list,axis=0)

        idxs=np.random.permutation(idxs)



        marginal_contribs = np.zeros(len(self.X))
        X_batch = np.zeros((0,) + tuple(self.X.shape[1:]))
        y_batch = np.zeros(0, int)
        sample_weight_batch = np.zeros(0)
        truncation_counter = 0
        new_score = self.random_score
        # self.restart_model()
        for n, idx in enumerate(idxs):
            old_score = new_score
            X_batch = np.concatenate([X_batch, self.X[sources[idx]]])
            y_batch = np.concatenate([y_batch, self.y[sources[idx]]])
            if self.sample_weight is None:
                sample_weight_batch = None
            else:
                sample_weight_batch = np.concatenate([
                    sample_weight_batch,
                    self.sample_weight[sources[idx]]
                ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if (self.is_regression
                    or len(set(y_batch)) == len(set(self.y_test))): ##FIXIT
                    self.restart_model()

                    num = len(X_batch)
                    x_g = np.arange(num)
                    mu = (num - 1) / 2
                    sigma = (num - 1) / 6
                    sample_weight_batch = np.exp(-1 * ((x_g - mu) ** 2) / (2 * (sigma ** 2))) / (
                                math.sqrt(2 * np.pi) * sigma)*num
                    # print(sample_weight_batch)
                    self.model.fit(
                        X_batch,
                        y_batch,
                        sample_weight = sample_weight_batch
                    )
                    new_score = self.value(self.model, metric=self.metric)
            marginal_contribs[sources[idx]] = (new_score - old_score)
            marginal_contribs[sources[idx]] /= len(sources[idx])
        return marginal_contribs, idxs

    def ourt_one_iteration(self, tolerance, sources=None):
        """Runs one iteration of our-Shapley algorithm."""
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}


        idxs = np.random.permutation(len(sources))


        y=self.y[idxs]
        class_list=set(y)
        idxs_list=[]
        for cs in class_list:

            tmp=idxs[y==cs]
            # print(cs, int(len(tmp) / 2))
            idxs_list.append(tmp[:int(len(tmp)*self.ratio)])
        idxs= np.concatenate(idxs_list,axis=0)

        idxs=np.random.permutation(idxs)



        marginal_contribs = np.zeros(len(self.X))
        X_batch = np.zeros((0,) + tuple(self.X.shape[1:]))
        y_batch = np.zeros(0, int)
        sample_weight_batch = np.zeros(0)
        truncation_counter = 0
        new_score = self.random_score
        # self.restart_model()
        for n, idx in enumerate(idxs):
            old_score = new_score
            X_batch = np.concatenate([X_batch, self.X[sources[idx]]])
            y_batch = np.concatenate([y_batch, self.y[sources[idx]]])
            if self.sample_weight is None:
                sample_weight_batch = None
            else:
                sample_weight_batch = np.concatenate([
                    sample_weight_batch,
                    self.sample_weight[sources[idx]]
                ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if (self.is_regression
                    or len(set(y_batch)) == len(set(self.y_test))): ##FIXIT
                    self.restart_model()

                    num = len(X_batch)
                    x_g = np.arange(num)
                    mu = (num - 1) / 2
                    sigma = (num - 1) / 6
                    sample_weight_batch = np.exp(-1 * ((x_g - mu) ** 2) / (2 * (sigma ** 2))) / (
                                math.sqrt(2 * np.pi) * sigma)*num
                    # print(sample_weight_batch)
                    self.model.fit(
                        X_batch,
                        y_batch,
                        sample_weight = sample_weight_batch
                    )
                    new_score = self.value(self.model, metric=self.metric)
            marginal_contribs[sources[idx]] = (new_score - old_score)
            marginal_contribs[sources[idx]] /= len(sources[idx])
            # print(n,new_score)
            distance_to_full_score = np.abs(new_score - self.mean_score)
            if distance_to_full_score <= tolerance * self.mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs

    def restart_model(self):
        
        try:
            self.model = clone(self.model)
        except:
            self.model.fit(np.zeros((0,) + self.X.shape[1:]), self.y)
        
    def _one_step_lr(self):
        """Computes the best learning rate for G-Shapley algorithm."""
        if self.directory is None:
            address = None
        else:
            address = os.path.join(self.directory, 'weights')
        best_acc = 0.0
        for i in np.arange(1, 5, 0.5):
            model = ShapNN(
                self.problem, batch_size=1, max_epochs=1, 
                learning_rate=10**(-i), weight_decay=0., 
                validation_fraction=0, optimizer='sgd', 
                warm_start=False, address=address, 
                hidden_units=self.hidden_units)
            accs = []
            for _ in range(10):
                model.fit(np.zeros((0, self.X.shape[-1])), self.y)
                model.fit(self.X, self.y)
                accs.append(model.score(self.X_test, self.y_test))
            if np.mean(accs) - np.std(accs) > best_acc:
                best_acc  = np.mean(accs) - np.std(accs)
                learning_rate = 10**(-i)
        return learning_rate
    
    def _g_shap(self, iterations, err=None, learning_rate=None, sources=None):
        """Method for running G-Shapley algorithm.
        
        Args:
            iterations: Number of iterations of the algorithm.
            err: Stopping error criteria
            learning_rate: Learning rate used for the algorithm. If None
                calculates the best learning rate.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        address = None
        if self.directory is not None:
            address = os.path.join(self.directory, 'weights')
        if learning_rate is None:
            try:
                learning_rate = self.g_shap_lr
            except AttributeError:
                self.g_shap_lr = self._one_step_lr()
                learning_rate = self.g_shap_lr
        model = ShapNN(self.problem, batch_size=1, max_epochs=1,
                     learning_rate=learning_rate, weight_decay=0.,
                     validation_fraction=0, optimizer='sgd',
                     address=address, hidden_units=self.hidden_units)
        for iteration in range(iterations):
            model.fit(np.zeros((0, self.X.shape[-1])), self.y)
            if 10 * (iteration+1) / iterations % 1 == 0:
                print('{} out of {} G-Shapley iterations'.format(
                    iteration + 1, iterations))
            marginal_contribs = np.zeros(len(sources.keys()))
            model.fit(self.X, self.y, self.X_test, self.y_test, 
                      sources=sources, metric=self.metric, 
                      max_epochs=1, batch_size=1)
            val_result = model.history['metrics']
            marginal_contribs[1:] += val_result[0][1:]
            marginal_contribs[1:] -= val_result[0][:-1]
            individual_contribs = np.zeros(len(self.X))
            for i, index in enumerate(model.history['idxs'][0]):
                individual_contribs[sources[index]] += marginal_contribs[i]
                individual_contribs[sources[index]] /= len(sources[index])
            self.mem_g = np.concatenate(
                [self.mem_g, np.reshape(individual_contribs, (1,-1))])
            self.idxs_g = np.concatenate(
                [self.idxs_g, np.reshape(model.history['idxs'][0], (1,-1))])
    
    def _calculate_loo_vals(self, sources=None, metric=None):
        """Calculated leave-one-out values for the given metric.
        
        Args:
            metric: If None, it will use the objects default metric.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        
        Returns:
            Leave-one-out scores
        """
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        print('Starting LOO score calculations!')
        if metric is None:
            metric = self.metric 
        self.restart_model()
        if 0:#self.sample_weight is None:
            self.model.fit(self.X, self.y)
        else:
            num = len(self.y)
            x_g = np.arange(num)
            mu = (num - 1) / 2
            sigma = (num - 1) / 6
            sw_batch = np.exp(-1 * ((x_g - mu) ** 2) / (2 * (sigma ** 2))) / (
                    math.sqrt(2 * np.pi) * sigma) * num
            self.model.fit(self.X, self.y,
                          sample_weight=self.sample_weight)
        baseline_value = self.value(self.model, metric=metric)
        vals_loo = np.zeros(len(self.X))
        for i in sources.keys():
            X_batch = np.delete(self.X, sources[i], axis=0)
            y_batch = np.delete(self.y, sources[i], axis=0)
            if self.sample_weight is not None:
                sw_batch = np.delete(self.sample_weight, sources[i], axis=0)
            if 0:#self.sample_weight is None:
                self.model.fit(X_batch, y_batch)
            else:
                num = len(X_batch)
                x_g = np.arange(num)
                mu = (num - 1) / 2
                sigma = (num - 1) / 6
                sw_batch = np.exp(-1 * ((x_g - mu) ** 2) / (2 * (sigma ** 2))) / (
                            math.sqrt(2 * np.pi) * sigma) * num
                self.model.fit(X_batch, y_batch, sample_weight=sw_batch)
                
            removed_value = self.value(self.model, metric=metric)
            # print(removed_value)
            vals_loo[sources[i]] = (baseline_value - removed_value)
            vals_loo[sources[i]] /= len(sources[i])
        return vals_loo
    
    def _merge_parallel_results(self, key, max_samples=None):
        """Helper method for 'merge_results' method."""
        numbers = [name.split('.')[-2].split('_')[-1]
                   for name in os.listdir(self.directory) 
                   if 'mem_{}'.format(key) in name]
        mem  = np.zeros((0, self.X.shape[0]))
        n_sources = len(self.X) if self.sources is None else len(self.sources)
        idxs = np.zeros((0, n_sources), int)
        vals = np.zeros(len(self.X))
        counter = 0.
        for number in numbers:
            if max_samples is not None:
                if counter > max_samples:
                    break
            samples_dir = os.path.join(
                self.directory, 
                'mem_{}_{}.pkl'.format(key, number)
            )
            print(samples_dir)
            dic = pkl.load(open(samples_dir, 'rb'))
            if not len(dic['mem_{}'.format(key)]):
                continue
            mem = np.concatenate([mem, dic['mem_{}'.format(key)]])
            idxs = np.concatenate([idxs, dic['idxs_{}'.format(key)]])
            counter += len(dic['mem_{}'.format(key)])
            vals *= (counter - len(dic['mem_{}'.format(key)])) / counter
            vals += len(dic['mem_{}'.format(key)]) / counter * np.mean(mem, 0)
            os.remove(samples_dir)
        merged_dir = os.path.join(
            self.directory, 
            'mem_{}_0000.pkl'.format(key)
        )
        pkl.dump({'mem_{}'.format(key): mem, 'idxs_{}'.format(key): idxs}, 
                 open(merged_dir, 'wb'))
        return mem, idxs, vals
            
    def merge_results(self, max_samples=None):
        """Merge all the results from different runs.
        
        Returns:
            combined marginals, sampled indexes and values calculated 
            using the two algorithms. (If applicable)
        """
        tmc_results = self._merge_parallel_results('tmc', max_samples)
        self.marginals_tmc, self.indexes_tmc, self.values_tmc = tmc_results
        if self.model_family not in ['logistic', 'NN']:
            return
        g_results = self._merge_parallel_results('g', max_samples)
        self.marginals_g, self.indexes_g, self.values_g = g_results
    
    def performance_plots(self, vals, name=None, 
                          num_plot_markers=20, sources=None, most=True):
        """Plots the effect of removing valuable points.
        
        Args:
            vals: A list of different valuations of data points each
                 in the format of an array in the same length of the data.
            name: Name of the saved plot if not None.
            num_plot_markers: number of points in each plot.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
                   
        Returns:
            Plots showing the change in performance as points are removed
            from most valuable to least.
        """
        plt.rcParams['figure.figsize'] = 8,8
        plt.rcParams['font.size'] = 25
        plt.xlabel('Fraction of train data removed (%)')
        plt.ylabel('Prediction accuracy (%)', fontsize=20)
        ratio=0.5
        if not isinstance(vals, list) and not isinstance(vals, tuple):
            vals = [vals]
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        vals_sources = [np.array([np.sum(val[sources[i]]) 
                                  for i in range(len(sources.keys()))])
                  for val in vals]
        if len(sources.keys()) < num_plot_markers:
            num_plot_markers = len(sources.keys()) - 1
        plot_points = np.arange(
            0, 
            max(len(sources.keys()) - 10, num_plot_markers),
            max(len(sources.keys())//num_plot_markers, 1)
        )
        if most:
            perfs = [self._portion_performance(
                np.argsort(vals_source)[::-1], plot_points, sources=sources)
                for vals_source in vals_sources]
        else:
            perfs = [self._portion_performance(
                np.argsort(vals_source), plot_points, sources=sources)
                for vals_source in vals_sources]

        rnd = np.mean([self._portion_performance(
            np.random.permutation(np.argsort(vals_sources[0])[::-1]),
            plot_points, sources=sources) for _ in range(10)], 0)


        plt.plot((plot_points / len(self.X) * 100)[:int(len(plot_points) * ratio)],
                 (perfs[0] * 100)[:int(len(plot_points) * ratio)],
                 '-', lw=5, ms=10, color='b')
        plt.plot((plot_points / len(self.X) * 100)[:int(len(plot_points) * ratio)],
                 (perfs[1] * 100)[:int(len(plot_points) * ratio)],
                 '-', lw=5, ms=10, color='g')
        plt.plot((plot_points / len(self.X) * 100)[:int(len(plot_points) * ratio)],
                 (perfs[2] * 100)[:int(len(plot_points) * ratio)],
                 '-', lw=5, ms=10, color='r')
        legends = ['CTMC-Shapley ', 'CMC-Shapley ', 'TMC-Shapley ']

        plt.legend(legends)
        if self.directory is not None and name is not None:
            plt.savefig(os.path.join(
                self.directory, 'plots', '{}.png'.format(name)),
                        bbox_inches = 'tight')
            plt.close()
        return perfs
            
    def _portion_performance(self, idxs, plot_points, sources=None):
        """Given a set of indexes, starts removing points from 
        the first elemnt and evaluates the new model after
        removing each point."""
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        scores = []
        init_score = self.random_score
        for i in range(len(plot_points), 0, -1):
            keep_idxs = np.concatenate([sources[idx] for idx 
                                        in idxs[plot_points[i-1]:]], -1)
            X_batch, y_batch = self.X[keep_idxs], self.y[keep_idxs]
            if self.sample_weight is not None:
                sample_weight_batch = self.sample_weight[keep_idxs]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if (self.is_regression 
                    or len(set(y_batch)) == len(set(self.y_test))):
                    self.restart_model()
                    if self.sample_weight is None:
                        self.model.fit(X_batch, y_batch)
                    else:
                        self.model.fit(X_batch, y_batch,
                                      sample_weight=sample_weight_batch)
                    scores.append(self.value(
                        self.model,
                        metric=self.metric,
                        X=self.X_heldout,
                        y=self.y_heldout
                    ))
                else:
                    scores.append(init_score)
        return np.array(scores)[::-1]
