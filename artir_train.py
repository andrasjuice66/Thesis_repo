# Imports
import random
import numpy as np
import CA_Utilities as cautils
import array
import sys
import ARTIR_ARTMAP_Class as artir_module
import torch
from avalanche.evaluation.metrics import Accuracy, BWT, Forgetting, ConfusionMatrix
import artir_data as data



def train_KFold(newf_count, num_new_features, artir_1, X, y, test_y, test_X, fam_baseline_vigilance, NC_ignore,
                acc_metric, e_results, min_fcor, cat_stats,
                artir_nfat, test_with_no_learn, epoch_results, average_weights, num_epochs, max_search_cycles):
    while newf_count <= num_new_features:
        artir_module.ARTIR.reset_all_cat_counts(artir_1)
        for train_index, test_index in data.K_Fold(X, 4):
            for tr in train_index:
                current_input = X[tr]
                current_target = y[tr]

                current_input_cc = cautils.ca_complement_code_elements(current_input)
                (search_history, ARTMAP_prediction, old_target) = \
                    artir_module.ARTIR.learn(artir_1, current_input_cc, fam_baseline_vigilance, current_target)

                # metrics
                # ======================
                if NC_ignore and ARTMAP_prediction != 'NC':
                    acc_target = torch.tensor([old_target])
                    acc_predict = torch.tensor([ARTMAP_prediction])
                    acc_metric.update(acc_target, acc_predict, 0)

                # ======================
            e_results.append(artir_1.count_used_cats())
            cat_stats[newf_count] += artir_1.count_used_cats()

        if newf_count < num_new_features:
            print("Searching for new feature (max cycles, min feature correlation, NFAT): ", max_search_cycles,
                  min_fcor, artir_nfat)
            (new_fmask, nfc, search_cycles) = artir_module.ARTIR.search_internal_feature(artir_1, max_search_cycles,
                                                                                         min_fcor)
            artir_module.ARTIR.add_new_feature(artir_1, new_fmask)

        if newf_count == num_new_features:
            print("Test phase")
            for te in test_index:
                current_input = X[te]
                current_target = y[te]

                current_input_cc = cautils.ca_complement_code_elements(current_input)
                if test_with_no_learn:
                    # expand current input with internal features before Fuzzy ARTMAP training cycle
                    new_features = artir_1.calc_features(current_input_cc)  # use object instance variable nfat
                    # expand current input with internal feature values
                    fam_input = current_input_cc + new_features
                    (search_history, match_level) = artir_1.recall(fam_input, fam_baseline_vigilance)
                    F2_winner = search_history[-1]
                    ARTMAP_prediction = artir_1.fam_classes[F2_winner]
                    print("Input:", current_input, "Target:", current_target, "Predicted:", ARTMAP_prediction,
                          "Correct:", (current_target == ARTMAP_prediction))
                    # metrics
                    # ======================
                    if NC_ignore and ARTMAP_prediction != 'NC':  # ignored NC-s
                        acc_target = torch.tensor([current_target])
                        acc_predict = torch.tensor([ARTMAP_prediction])
                        acc_metric.update(acc_target, acc_predict, 1)
                    # ======================

                    artir_1.incr_cat_count(F2_winner)
                    if current_target != ARTMAP_prediction:
                        artir_1.nr_wrong += 1
                else:
                    (search_history, ARTMAP_prediction, old_target2) = \
                        artir_module.ARTIR.learn(artir_1, current_input_cc, fam_baseline_vigilance, current_target)

                    # metrics
                    # ======================
                    if NC_ignore and ARTMAP_prediction != 'NC':  # ignored NC-s
                        acc_target = torch.tensor([old_target2])
                        acc_predict = torch.tensor([ARTMAP_prediction])
                        acc_metric.update(acc_target, acc_predict, 2)
                    # ======================

                e_results.append(artir_1.count_used_cats())
                cat_stats[newf_count] += artir_1.count_used_cats()
        newf_count += 1
        epoch_results.append(e_results)
        average_weights += len(artir_1.fam_weights) - 1
    print("Number of new features: ", newf_count)
    print("Average weight number: ", average_weights / num_epochs)
    print("Accuracy: ", acc_metric.result())
    print("=========== END ================")

def train_normal_mnist(newf_count, num_new_features, artir_1, X, y, test_y, test_X, fam_baseline_vigilance, NC_ignore,
                 acc_metric,
                 e_results, min_fcor, cat_stats,
                 artir_nfat, test_with_no_learn, epoch_results, average_weights, num_epochs, max_search_cycles,
                 num_task, instance_per_task):
    average_weights = 0
    acc_metric = Accuracy()
    bwt_metric = BWT()
    forgetting_metric = Forgetting()
    confusion_matrix_metric = ConfusionMatrix(4, normalize="true")
    task_label = 0
    for i in range(num_task):
        task_label +=1

        while (newf_count) <= num_new_features:
            artir_module.ARTIR.reset_all_cat_counts(artir_1)

            for tr in y[(i * instance_per_task): ((i + 1) * instance_per_task)]:
                current_input = X[tr]
                current_target = y[tr]

                current_input_cc = cautils.ca_complement_code_elements(current_input)
                (search_history, ARTMAP_prediction, old_target) = \
                    artir_module.ARTIR.learn(artir_1, current_input_cc, fam_baseline_vigilance, current_target)

                # metrics
                # ======================
                if NC_ignore and ARTMAP_prediction != 'NC':
                    acc_target = torch.tensor([old_target])
                    acc_predict = torch.tensor([ARTMAP_prediction])
                    acc_metric.update(acc_target, acc_predict, task_label)

                # ======================
            e_results.append(artir_1.count_used_cats())
            cat_stats[newf_count] += artir_1.count_used_cats()

            if newf_count < num_new_features:
                print("Searching for new feature (max cycles, min feature correlation, NFAT): ", max_search_cycles,
                      min_fcor, artir_nfat)
                (new_fmask, nfc, search_cycles) = artir_module.ARTIR.search_internal_feature(artir_1, max_search_cycles,
                                                                                             min_fcor)
                artir_module.ARTIR.add_new_feature(artir_1, new_fmask)

            if newf_count == num_new_features:
                print("Test phase")
                task_label += 1

                for te in test_y[(i * instance_per_task): ((i + 1) * instance_per_task)]:

                    current_input = test_X[te]
                    current_target = test_y[te]

                    current_input_cc = cautils.ca_complement_code_elements(current_input)
                    if test_with_no_learn:
                        # expand current input with internal features before Fuzzy ARTMAP training cycle
                        new_features = artir_1.calc_features(current_input_cc)  # use object instance variable nfat
                        # expand current input with internal feature values
                        fam_input = current_input_cc + new_features
                        (search_history, match_level) = artir_1.recall(fam_input, fam_baseline_vigilance)
                        F2_winner = search_history[-1]
                        ARTMAP_prediction = artir_1.fam_classes[F2_winner]
                        print("Input:", current_input, "Target:", current_target, "Predicted:", ARTMAP_prediction,
                              "Correct:", (current_target == ARTMAP_prediction))
                        # metrics
                        # ======================
                        if NC_ignore and ARTMAP_prediction != 'NC':  # ignored NC-s
                            acc_target = torch.tensor([current_target])
                            acc_predict = torch.tensor([ARTMAP_prediction])
                            acc_metric.update(acc_target, acc_predict, task_label)
                        # ======================

                        artir_1.incr_cat_count(F2_winner)
                        if current_target != ARTMAP_prediction:
                            artir_1.nr_wrong += 1
                    else:
                        (search_history, ARTMAP_prediction, old_target2) = \
                            artir_module.ARTIR.learn(artir_1, current_input_cc, fam_baseline_vigilance, current_target)

                        # metrics
                        # ======================
                        if NC_ignore and ARTMAP_prediction != 'NC':  # ignored NC-s
                            acc_target = torch.tensor([old_target2])
                            acc_predict = torch.tensor([ARTMAP_prediction])
                            acc_metric.update(acc_target, acc_predict, 2)
                        # ======================

                    e_results.append(artir_1.count_used_cats())
                    cat_stats[newf_count] += artir_1.count_used_cats()
            newf_count += 1
            epoch_results.append(e_results)
            average_weights += len(artir_1.fam_weights) - 1
        newf_count = 0 # initialize at every new task

    print("Number of new features: ", newf_count)
    print("Average weight number: ", average_weights / num_epochs)
    print("Accuracy: ", acc_metric.result())
    print("=========== END ================")