# Imports
import random
import numpy as np
import CA_Utilities as cautils
import sys
import ARTIR_ARTMAP_Class as artir_module
import torch
from avalanche.evaluation.metrics import Accuracy, BWT, Forgetting, ConfusionMatrix
import artir_data as data
import artir_train as train

# PARAMETERS
fam_baseline_vigilance = 0.1
min_fcor = 0.6
max_search_cycles = 100
num_training_cycles = 5
num_new_features = 2
artir_nfat = 0.7

num_epochs = 1

# CONTROLS
test_with_no_learn = True
NC_ignore = True
isKFOLD = False
isBinary = False

# METRICS
average_weights = 0
acc_metric = Accuracy()
bwt_metric = BWT()
forgetting_metric = Forgetting()
confusion_matrix_metric = ConfusionMatrix(4, normalize="true")

# DATASET
(X, y, test_X, test_y) = data.choose_dataset('iris')

# CREATE SCENARIO
(num_task, instance_per_task) = data.create_scenario(isBinary, isKFOLD, X, y, test_X, test_y)

# INITIALIZE
artir_1 = artir_module.ARTIR()
(artir_1, cat_stats, fkernel_list) = artir_module.ARTIR.new_instance(artir_1, X, num_new_features, artir_nfat, False)
newf_count = 0

epoch_results = [
    [fam_baseline_vigilance, min_fcor, max_search_cycles, num_training_cycles, num_new_features,
     artir_nfat]]

for epochs in range(num_epochs):
    e_results = [epochs]

    print("Epoch: ", epochs, " ==============================================================")

    if isKFOLD:
        train.train_KFold(newf_count, num_new_features, artir_1, X, y, test_y, test_X, fam_baseline_vigilance,
                          NC_ignore, acc_metric,
                          e_results, min_fcor, cat_stats,
                          artir_nfat, test_with_no_learn, epoch_results, average_weights, num_epochs, max_search_cycles)
    else:
        train.train_normal_mnist(newf_count, num_new_features, artir_1, X, y, test_y, test_X, fam_baseline_vigilance,
                           NC_ignore, acc_metric,
                           e_results, min_fcor, cat_stats,
                           artir_nfat, test_with_no_learn, epoch_results, average_weights, num_epochs,
                           max_search_cycles, num_task, instance_per_task)



sys.exit()
