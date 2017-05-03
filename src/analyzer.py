#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The data processor running autoencoders training

@author: yaric
"""

import os
import sys
import timeit
import argparse

import numpy as np

import cA
import dA

import theano
import theano.tensor as T

def train(data, batch_size, learning_rate = 0.01, contraction_level = .1, corruption_level = 0.3, 
            n_hidden = 7, training_epochs = 100, encoder = 'cA'):
    """
    The method to run autoencoder training against provided data with specified
    encoder type
    Arguments:
        data the the input data array
        batch_size the number of samples per batch
        learning_rate the learning rate
        contraction_level the contraction level for contractive auto-encoder
        corruption_level the corruption level to apply to input data for denoising auto-encoder
        n_hidden the number of hidden layer units
        training_epochs the number of training epochs
        encoder the auto-encoder type (cA, dA)[default: cA]
    Return:
        (W, costs) the tuple with learned encoder's hidden layer weights and
        training costs (errors) per train epoch
    """
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    
    rng = np.random.RandomState(42)
    
    if encoder == 'cA':
        ca = cA(numpy_rng = rng, input = x,
            n_visible = data.shape[1], n_hidden = n_hidden, n_batchsize = batch_size)

        cost, updates = ca.get_cost_updates(contraction_level = contraction_level,
                                            learning_rate = learning_rate)
    
        train = theano.function(
            [index],
            [T.mean(ca.L_rec), ca.L_jacob],
            updates = updates,
            givens = {
                x: data[index * batch_size: (index + 1) * batch_size]
            }
        )
    elif encoder == 'dA':
        da = dA(numpy_rng = rng, input = x,
            n_visible = data.shape[1], n_hidden = n_hidden
        )
    
        cost, updates = da.get_cost_updates(
            corruption_level = 0.3,
            learning_rate = learning_rate
        )
    
        train = theano.function(
            [index],
            cost,
            updates = updates,
            givens = {
                x: data[index * batch_size: (index + 1) * batch_size]
            }
        )
    else:
        raise Exception('Unknown auto-encoder type requested: ' + encoder)
    
    #
    # do training
    #
    start_time = timeit.default_timer()
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = data.get_value(borrow=True).shape[0] // batch_size
    
    # go through training epochs
    epoch_costs = []
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            b_cost = train(batch_index) # the batch cost
            c.append(b_cost)
            if encoder == 'cA':
                epoch_costs.append(np.mean(b_cost[0]))
            elif encoder == 'dA':
                epoch_costs.append(np.mean(b_cost))

        c_array = np.vstack(c)
        if encoder == 'cA':
            print('Training epoch %d, reconstruction cost ' % epoch, np.mean(
                c_array[0]), ' jacobian norm ', np.mean(np.sqrt(c_array[1])))
        elif encoder == 'dA':
            print('Training epoch %d, cost ' % epoch, np.mean(c_array, dtype='float64'))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print(('The code for file ' + os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((training_time) / 60.)), file=sys.stderr)
    
    if encoder == 'cA':
        return (ca.W.get_value(borrow=True).T, epoch_costs)
    elif encoder == 'dA':
        return (da.W.get_value(borrow=True).T, epoch_costs)

def analyse(input_file, **args):
    """
    Performs input data analysis
    """
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The input data preprocessor')