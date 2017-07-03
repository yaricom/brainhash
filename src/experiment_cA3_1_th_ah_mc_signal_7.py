#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The experiment with 10 Hz/5Hz, wisp, attention, 70, cA 3, theta, alpha high, batch size = 1 and 
multiclass data set (BALANCED) with signal only data

@author: yaric
"""

import experiment as ex
import config
from time import time

n_hidden = 3
batch_size = 1
max_cls_samples = 7

experiment_name = 'cA_%d_%d_th-a_h_mc_signal_%d' % (n_hidden, batch_size, max_cls_samples) # will be used as parent dir for analyzer results

# The sample records identifiers
signal_ids = ['IO_10_2', 'KS_10_2', 'RO_10_2']
class_lbs = ['IO', 'KS', 'RO']

# Setup analyzer configuration
analyzer_config = ex.defaultAnalyzerConfig()
analyzer_config['batch_size']       = batch_size
analyzer_config['learning_rate']    = 0.1
analyzer_config['n_hidden']         = n_hidden
analyzer_config['training_epochs']  = 50000
analyzer_config['encoder']          = 'cA'
analyzer_config['bands']            = 'theta,alpha_h'

start = time()

#
# Run analyzer
#
print("\nStart analysis with parameters:\n%s\n" % analyzer_config)
print("Start analysis for signal records: %s" % signal_ids)

ex.runEEGAnalyzerWithIDs(ids_list=signal_ids, 
                         experiment_name=experiment_name,
                         a_config=analyzer_config)



#
# Run classifiers
#
#analyzed_res_dir = experiment_name
analyzed_res_dir = experiment_name
signal_dir  = "%s/%s" % (config.analyzer_out_dir, analyzed_res_dir)
out_suffix  = experiment_name
print("Run classifiers over analyzed records. \nSignal dir: %s" 
      % (signal_dir))

ex.runSignalsOnlyClassifier(signal_dir=signal_dir, 
                 signal_records=signal_ids, 
                 out_suffix=out_suffix,
                 signal_class_labels=class_lbs,
                 max_cls_samples=max_cls_samples)

print("\n\nExperiment %s took %.2f seconds.\n"
          % (experiment_name, time() - start))