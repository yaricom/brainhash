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
import matplotlib.pyplot as plt

from cA import cA
from dA import dA

import theano
import theano.tensor as T

import config
import utils

# the names of bands to be accepted
bands_names = ['delta', 'theta', 'alpha_l', 'alpha_h', 'beta_l', 'beta_h', 'gamma_l', 'gamma_h']

def train(data, batch_size, n_visible, n_hidden = 7, learning_rate = 0.01, 
          contraction_level = .1, corruption_level = 0.3, 
          training_epochs = 100, encoder = 'cA'):
    """
    The method to run autoencoder training against provided data with specified
    encoder type
    Arguments:
        data the the input data array
        batch_size the number of samples per batch
        n_visible the number of input units
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
    x = T.matrix('x')  # the data 
    
    rng = np.random.RandomState(42)
    
    if encoder == 'cA':
        ca = cA(numpy_rng = rng, input = x,
            n_visible = n_visible, n_hidden = n_hidden, n_batchsize = batch_size)

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
            n_visible = n_visible, n_hidden = n_hidden
        )
    
        cost, updates = da.get_cost_updates(
            corruption_level = corruption_level,
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
            c.append(train(batch_index))

        c_array = np.vstack(c)
        if encoder == 'cA':
            cost = np.mean(c_array[:,0], dtype='float64')
            if epoch % 100 == 0:
                print('Training epoch %d, reconstruction cost ' % epoch, cost, 
                      ' jacobian norm ', np.mean(np.sqrt(c_array[:,1])))
            
        elif encoder == 'dA':
            cost = np.mean(c_array, dtype='float64')
            if epoch % 100 == 0:
                print('Training epoch %d, cost ' % epoch, cost)
            
        # store epoch cost
        epoch_costs.append(cost)

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print(('The code for file ' + os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((training_time) / 60.)), file=sys.stderr)
    
    if encoder == 'cA':
        return (ca.W.get_value(borrow=True).T, epoch_costs)
    elif encoder == 'dA':
        return (da.W.get_value(borrow=True).T, epoch_costs)

def analyse(input_file, out_file, batch_size, learning_rate, contraction_level,
            corruption_level, n_hidden, training_epochs, encoder, in_bands, save_plot):
    """
    Performs input data analysis
    """
    print("Start analysis of: %s and saving scores to: %s" % (input_file, out_file))
    
    # load input data array
    data = np.load(input_file)
    
    if in_bands != 'all':
        indx = []
        for name in in_bands.split(','):
            indx.append(bands_names.index(name))
            print("%s -> %d" % (name,bands_names.index(name)))
        data = data[:, indx]
    
    # set batch size
    if batch_size == None:
        batch_size = 1
    else:
        batch_size = batch_size
        
    print("The batch size: %d" % batch_size)
    
    shared_data = theano.shared(np.asarray(data, dtype=theano.config.floatX), 
                                borrow = True)
    
    scores, costs = train(shared_data, batch_size = batch_size, 
                          n_visible = data.shape[1], n_hidden = n_hidden, 
                          learning_rate = learning_rate,
                          contraction_level = contraction_level,
                          corruption_level = corruption_level,
                          training_epochs = training_epochs,
                          encoder = encoder)
    
    # check output directory
    utils.checkParentDir(out_file, clear = False)
    np.save(out_file, scores)
    
    # plot results skipping first values
    plt.figure()
    plt.title('Costs per epoch')
    plt.xlabel('epoch')
    plt.ylabel('cost')
    x = np.arange(100, training_epochs)
    plt.plot(x, costs[100:training_epochs], 'r-')

    # save plot
    if save_plot:
        plt.savefig(out_file + ".png", format='png')
        
    # close plot
    plt.close()
    
    # print formatted final results
    print("Reconstruction cost: ", costs[-1])
    if encoder == 'cA':
        print("learning_rate: %.4f, batch_size: %d, training_epochs: %d, n_hidden: %d, contraction_level: %.2f, encoder: %s\n" %
              (learning_rate, batch_size, training_epochs, n_hidden, contraction_level, encoder))
    elif encoder == 'dA':
        print("learning_rate: %.4f, batch_size: %d, training_epochs: %d, n_hidden: %d, corruption_level: %.2f, encoder: %s\n" %
              (learning_rate, batch_size, training_epochs, n_hidden, corruption_level, encoder))
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The input data preprocessor')
    parser.add_argument('input_file', default = config.preprocessor_out_file,  
                        help='the input data file as Numpy array')
    parser.add_argument('--out_file', default = config.analyzer_scores_out_file, 
                        help='the file to store analyzer scores output')
    parser.add_argument('--batch_size', type=int,
                        help='the number of samples per batch')
    parser.add_argument('--learning_rate', default = 0.1, type=float,
                        help='the learning rate')
    parser.add_argument('--contraction_level', default = 0.1, type=float,
                        help='the contraction level for contractive auto-encoder')
    parser.add_argument('--corruption_level', default = 0.1, type=float,
                        help='the corruption level to apply to input data for denoising auto-encoder')
    parser.add_argument('--n_hidden', default = 16, type=int,
                        help='the number of hidden layer\'s units')
    parser.add_argument('--training_epochs', default = 10000, type=int, #50000, 
                        help='the number of training epochs')
    parser.add_argument('--encoder', default = 'cA',
                        help='the auto-encoder type (cA, dA)')
    parser.add_argument('--save_plot', default = True, type=bool,
                        help='Flag to indicate whether to save train plot to file')
    parser.add_argument('--bands', default = 'all',
                        help='The names of EEG bands to process')
    args = parser.parse_args()
    
    analyse(input_file = args.input_file, 
            out_file = args.out_file,
            batch_size = args.batch_size, 
            learning_rate = args.learning_rate,
            contraction_level = args.contraction_level,
            corruption_level = args.corruption_level,
            n_hidden = args.n_hidden,
            training_epochs = args.training_epochs,
            encoder = args.encoder,
            save_plot = args.save_plot,
            in_bands = args.bands)