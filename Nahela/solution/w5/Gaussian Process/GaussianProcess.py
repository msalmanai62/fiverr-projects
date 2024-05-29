
import sys
import time
import numpy as np
from scipy.optimize import minimize
from gaussian_processes_util import plot_gp
from gaussian_processes_util import nll_fn
from gaussian_processes_util import posterior
from sklearn import metrics
from ModelEvaluator import ModelEvaluator


class GaussianProcess():
    noise = 0.4
    mu = None # mean vector to be estimated
    cov = None # covariance matrix to be estimated
    predictions = [] # probabilities of test data points
    running_time = None # execution time for training+test
    baseline_variant1 = False # use baseline discussed in lecture

    def __init__(self, datafile_train, datafile_test):
        # Load training and test data from two separate CVS files
        X_train, Y_train = self.loadCVSFile(datafile_train)
        X_test, Y_test = self.loadCVSFile(datafile_test)
		
		# train GP model via regression and evaluate it with test data
        self.running_time = time.time()
        self.estimate_mean_and_covariance(X_train, Y_train, X_test)
        self.running_time = time.time() - self.running_time
        self.evaluate_model_baseline(X_test, Y_test)

    def loadCVSFile(self, fileName):
        print("LOADING CSV file %s" % (fileName))
        X = []
        Y = []
        start_reading = False
        with open(fileName) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not start_reading:
                    # ignore the header/first line
                    start_reading = True
                else:
                    values = [float(i) for i in line.split(',')]
                    X.append(values[:len(values)-1])
                    Y.append(values[len(values)-1:])
        return np.asarray(X), np.asarray(Y)

    def estimate_mean_and_covariance(self, X_train, Y_train, X_test):
        print("ESTIMATING mean and covariance for GP model...")

        # search for optimal values for l and sigma_f via negative log-likelihood
        # and the Limited-memory BGGS-B algorithm. The latter isan extension of 
        # the L-BFGS algorithm used for hyperparameter optimisation
        res = minimize(nll_fn(X_train, Y_train, self.noise, False), [1, 1], 
                       bounds=((1e-5, None), (1e-5, None)), method='L-BFGS-B')
        l_opt, sigma_f_opt = res.x

        print("Hyperparameters: l=%s sigma=%s noise=%s" % (l_opt, sigma_f_opt, self.noise))
		
        # Compute posterior mean and covariance using optimised kernel parameters
        self.mu, self.cov = posterior(X_test, X_train, Y_train, \
                            l=l_opt, sigma_f=sigma_f_opt, sigma_y=self.noise)

    def get_gaussian_probability_density(self, x, mean, var):
        e_val = -np.power((x-mean), 2)/(2*var)
        return (1/(np.sqrt(2*np.pi*var))) * np.exp(e_val)

    def evaluate_model_baseline(self, X_test, Y_test):
        mu = self.mu.reshape(-1, 1)
        var = 1.96 * np.sqrt(np.diag(self.cov)).reshape(-1, 1)
        _min = np.min(mu)
        _max = np.max(mu)
        Y_true = Y_test
        Y_pred = []
        Y_prob = []

        for i in range(0, len(X_test)):
            if self.baseline_variant1: 
                prob = float((mu[i]-_min)/(_max-_min))
            else:
                pdf_1 = self.get_gaussian_probability_density(1, self.mu[i], var[i])
                pdf_0 = self.get_gaussian_probability_density(0, self.mu[i], var[i])
                prob = pdf_1 / (pdf_1 + pdf_0)
            Y_prob.append(prob)

            print("[%s] X_test=%s Y_test=%s mu=%s var=%s p=%s" % \
			      (i, X_test[i], Y_test[i], self.mu[i], var[i], prob))
            if prob >= 0.5:
                Y_pred.append(1)
            else:
                Y_pred.append(0)

        ModelEvaluator.compute_performance(None, None, Y_true, Y_pred, Y_prob)
        print("Running Time="+str(self.running_time)+" secs.")



datafile_train = '../w5/dataset/diabetes_data-discretized-train.csv'
datafile_test = "../w5/dataset/diabetes_data-discretized-test.csv"
GaussianProcess(datafile_train, datafile_test)
