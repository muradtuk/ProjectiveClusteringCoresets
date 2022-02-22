"""*****************************************************************************************
MIT License
Copyright (c) 2022 Murad Tukan, Xuan Wu, Samson Zhou, Vladimir Braverman, Dan Feldman
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*****************************************************************************************"""


import numpy as np
import time
import copy
import PointSet


class Coreset(object):
    """
    ################## Coreset ####################
    Functions:
        - __init__ : instructor
        - computeCoreset
        - mergeCoreset
    """

    def __init__(self, prob_dep_vars=None, is_uniform=False):
        """

        :param prob_dep_vars: Problem dependant variables (not used)
        :param is_uniform: A boolean variable stating whether uniform or importance sampling, i.e., using sensitivity
        (Default value: False)
        """
        self.weights = []
        self.S = []
        self.probability = []
        self.is_uniform = is_uniform
        self.prob_dep_vars = prob_dep_vars

    def computeCoreset(self, P, sensitivity, sampleSize, weights=None, SEED=1.0, is_uniform=True):
        """
        :param P: A matrix of nxd points where the last column is the labels.
        :param sensitivity: A vector of n entries (number of points of P) which describes the sensitivity of each point.
        :param sampleSize: An integer describing the size of the coreset.
        :param weights: A weight vector of the data points (Default value: None)
        :return: A subset of P (the datapoints alongside their respected labels), weights for that subset and the
        time needed for generating the coreset.
        """

        startTime = time.time()
        sens = copy.deepcopy(sensitivity)

        # Compute the sum of sensitivities.
        t = np.sum(sensitivity)

        # The probability of a point prob(p_i) = s(p_i) / t
        self.probability = sensitivity.flatten() / t

        avoid_big_weights = np.where(self.probability >= 1.0 / sampleSize)[0]

        if avoid_big_weights.size == 0 or is_uniform or True:
            S = None
            Q = copy.deepcopy(P.P)
            W = copy.deepcopy(P.W)
            weights = None
        else:
            remaining_points = np.setdiff1d(np.array(list(range(P.n))), avoid_big_weights)
            S = copy.deepcopy(P.P[avoid_big_weights, :])
            weights = copy.deepcopy(P.W[avoid_big_weights])
            Q = copy.deepcopy(P.P)
            W = copy.deepcopy(P.W)
            sens[avoid_big_weights] = 0.0
            self.probability = sens / np.sum(sens)

        # The number of points is equivalent to the number of rows in P.
        n = P.n

        # initialize new seed
        # np.random.seed()

        # # Multinomial distribution.
        indxs = np.random.choice(n, sampleSize, p=self.probability.flatten())
        #
        # # Compute the frequencies of each sampled item.
        hist = np.histogram(indxs, bins=range(n))[0].flatten()
        indxs = copy.deepcopy(np.nonzero(hist)[0])

        hist = np.random.multinomial(sampleSize, self.probability.flatten()).flatten()
        # print('hist {}'.format(hist))
        # print('hist.shape {}'.format(hist.shape))
        indxs = np.nonzero(hist)[0]

        # Select the indices.
        if S is None:
            S = Q[indxs, :]
        else:
            S = np.vstack((S, Q[indxs, :]))

        # Compute the weights of each point: w_i = (number of times i is sampled) / (sampleSize * prob(p_i))
        weights_of_remaining = np.asarray(np.multiply(W[indxs], hist[indxs]), dtype=float).flatten()

        # Compute the weights of the coreset
        weights_of_remaining = np.multiply(weights_of_remaining, 1.0 / (self.probability[indxs]*sampleSize))
        if weights is not None:
            weights = np.hstack((weights, weights_of_remaining))
        else:
            weights = weights_of_remaining

        weights = weights #* np.sum(P.W) / np.sum(weights)

        timeTaken = time.time() - startTime
        S = PointSet.PointSet(S, weights)

        return S, timeTaken

    @staticmethod
    def mergeCoreset(coreset_1, coreset_2):
        coreset_1.rank += 1
        coreset_1.merge(coreset_2)
        return