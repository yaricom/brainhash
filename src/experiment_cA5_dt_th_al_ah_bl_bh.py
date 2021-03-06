#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The experiment with 10 Hz/5Hz, wisp, attention, 70, cA 5, delta, theta, alpha low, alpha high, beta low, beta high, batch size = 5 and 
balanced data set

@author: yaric
"""

import experiment as ex
import config
from time import time

experiment_name = 'cA_5_dt-th-a_l-a_h-b_l-b_h' # will be used as parent dir for analyzer results

# The sample records identifiers
signal_ids = ['IO_10_2', 'IO_TXT', 'IO_SKY', 'KS_10_2', 'RO_10_2']
noise_ids  = ['noise']

# Setup analyzer configuration
analyzer_config = ex.defaultAnalyzerConfig()
analyzer_config['batch_size']       = 5
analyzer_config['learning_rate']    = 0.1
analyzer_config['n_hidden']         = 5
analyzer_config['training_epochs']  = 50000
analyzer_config['encoder']          = 'cA'
analyzer_config['bands']            = 'delta,theta,alpha_l,alpha_h,beta_l,beta_h'

start = time()

#
# Run analyzer
#
print("\nStart analysis with parameters:\n%s\n" % analyzer_config)
print("Start analysis for signal records: %s" % signal_ids)
ex.runEEGAnalyzerWithIDs(ids_list=signal_ids, 
                         experiment_name=experiment_name,
                         a_config=analyzer_config)

print("Start analysis for noise records: %s" % noise_ids)
ex.runEEGAnalyzerWithIDs(ids_list=noise_ids, 
                         experiment_name=experiment_name,
                         a_config=analyzer_config)


#
# Run classifiers
#
signal_dir  = "%s/%s" % (config.analyzer_out_dir, experiment_name)
noise_dir   = "%s/%s/%s" % (config.analyzer_out_dir, experiment_name, noise_ids[0])
out_suffix  = experiment_name
print("Run classifiers over analyzed records. \nSignal dir: %s\nNoise dir: %s" 
      % (signal_dir, noise_dir))
ex.runClassifier(signal_dir=signal_dir, 
                 signal_records=signal_ids, 
                 noise_dir=noise_dir, 
                 out_suffix=out_suffix)

print("\n\nExperiment %s took %.2f seconds.\n"
          % (experiment_name, time() - start))

