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
import copy


class PointSet(object):
    """
    This class is dedicated for implementing the notion of the weighted set of points.

    Functions:
                - __init__: This is the constructor of the class.
                - attainSubset: This functions returns a subset of the input points of the class (self.P)

    """
    def __init__(self, P, W=None):
        """
        This function constructs an instance of the class.

        :param P: A numpy matrix of points.
        :param W:  A numpy array of corresponding weights (with respect to the rows of P). Default value is None.
        """
        self.P = P
        self.n, self.d = P.shape  # number of points, the dimension of the points

        if W is None:  # if no weights were given
            W = np.ones(self.n)  # set unit weights

        self.W = W

    def attainSubset(self, idxs):
        """
        This functions returns a PointSet instance which contans a subset of the class input weighted set (self.P).

        :param idxs: A numpy array of indices of points of self.P.P
        :return: A PointSet instance which contains the desires subset of points chosen by idsx.
        """
        P = copy.deepcopy(self.P[idxs, :])  # attain subset of the points chosen by idxs
        W = copy.deepcopy(self.W[idxs])  # attain subset of the corresponding weights chosen by idxs

        return PointSet(P, W)

    def reduceDimension(self, X, include_last_column=False):
        """
        This function is used for reducing the dimension of the points by taking the dot product between each point and
        a subspace.

        :param X: An orthogonal matrix denoting a linear subspace
        :param include_last_column: An indicator variable used for distinguishing whether the last entry of each point
                                    is used for the index of that point.
        :return: None.
        """
        self.P = np.hstack((self.P[:, :-1].dot(X), self.P[:, -1][:, np.newaxis])) if not include_last_column else \
            self.P.dot(X)
        self.n, self.d = self.P.shape

    def unionPointSets(self, P):
        """
        The function at hand, unifies between two weighted sets of points (the current class denoted by self and P)

        :param P: A PointSet object, i.e., a weighted set of points.
        :return: None.
        """
        self.P = np.vstack((self.P, P.P))  # inserting the points in P into the current set of points (self.P)
        self.W = np.hstack((self.W, P.W))  # inserting the corresponding weights of points in P to current list of
                                           # weights (self.W)
        u, indices = np.unique(self.P, axis=0, return_index=True)  # remove duplicated points
        self.P = self.P[indices, :]
        self.W = self.W[indices]
        self.n, self.d = self.P.shape  # update the number of points and the dimension of the points

    def addIndices(self):
        self.P = np.hstack((self.P, np.arange(self.n)[:, np.newaxis]))

    def __len__(self):
        return self.n
