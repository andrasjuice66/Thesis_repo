# -*- coding: utf-8 -*-
"""

'Creative ART' (ARTIR) implementation as a Python class

Author: Gusztav Bartfai
Date: Feb 5, 2021

Description:
    Implementation of 'Creative ART' (ARTIR) algorithm based on Fuzzy ARTMAP (Fuzzy ARTMAP Class)
    Class functions defined:
        - __init__
        - initialize
        - set_nfat
        - set_kernels
        - set_fmethod
        - calc_features
        - add_new_feature
        - learn
        # - recall
        - calc_feature_correlation
        - prune_fmask_categories
        - calc_feature_value_generic (static)
        - calc_feature_value_basic (static)
        - calc_feature_value_kernel (static)
        - gen_feature_mask_cc (static)
        - gen_feature_mask_cc_simple (static)

Note:
    ...

"""

# Imports
import random

import numpy as np
#import matplotlib.pyplot as plt
import CA_Utilities as cautils
#import Fuzzy_ARTMAP_Class as fam_class
import torch
from avalanche.evaluation.metrics import Accuracy


# Parameters

ARTIR_NF_NOF2 = "NoF2"
ARTIR_NEW_FEATURE_MASK_INIT_VALUE = 1
ARTIR_NFAT_DEFAULT_VALUE = 0.5
ARTIR_KERNELS_DEFAULT_LIST = [[0, 0], [1, 1]]
ARTIR_FEATURE_METHOD_BASIC = 'Basic'  # the basic method with feature mask and NFAT (i.e. feature_threshold)
ARTIR_FEATURE_METHOD_KERNEL = 'Kernel'  # the kernel based feature calculation


# ARTIR class as a child of Fuzzy ARTMAP Python class

FAM_WEIGHT_INIT_VALUE = 1
FAM_NOCLASS_VALUE = "NC"
FAM_MT_VIGILANCE_INCREASE = 0.001  # by how much ARTa vigilance should exceed the current matching level during match tracking
FAM_CHOICE_ALPHA_DEFAULT_VALUE = 0.001
FAM_LEARNING_RATE_DEFAULT_VALUE = 1  # Fast learning mode


class Fuzzy_ARTMAP:
    # class variables
    fam_learning_rate = FAM_LEARNING_RATE_DEFAULT_VALUE  # default learning rate for the whole FAM class
    fam_choice_param = FAM_CHOICE_ALPHA_DEFAULT_VALUE  # default choice parameter (alpha)

    # class constructor
    def __init__(self):
        self.fam_weights = []  # F2 weight vectors
        self.fam_classes = []  # Class association of each F2 node (simplified map field)
        self.fam_catcount = []  # count how many times each F2 node (category) has been selected since counter reset

    # initialize Fuzzy ARTMAP data structures
    def initialize(self, dim_input):
        self.fam_weights = [
            [FAM_WEIGHT_INIT_VALUE for i in range(dim_input)]]  # single F2 node, weights are all 1 initially
        self.fam_classes = [FAM_NOCLASS_VALUE]  # no class for the single F2 node
        self.fam_catcount = [0]  # set category counter for single F2 node to zero

    # add a new F2 node with initial values
    # it assumes that there as at least one existing F2 node
    def add_F2_node(self):
        dim_input = len(self.fam_weights[0])
        new_F2_node = [FAM_WEIGHT_INIT_VALUE for i in range(dim_input)]
        self.fam_weights.append(new_F2_node)
        self.fam_classes.append(FAM_NOCLASS_VALUE)
        self.fam_catcount.append(0)

    # reset given F2 node and its class mapping
    def reset_F2_node(self, F2_to_reset):
        dim_F2_node = len(self.fam_weights[F2_to_reset])
        self.fam_weights[F2_to_reset] = [0 for i in range(dim_F2_node)]  # set all weights for this F2 node to zero
        self.fam_classes[F2_to_reset] = FAM_NOCLASS_VALUE  # this F2 node has no longer any class mapping

    # reset all category counters
    def reset_all_cat_counts(self):
        self.fam_catcount = [0 for i in range(len(self.fam_weights))]  # set all category node counters to zero

    # increment a given F2 node (category) counter
    # useful for counting categories for ARTMAP recall function only (which does not change F2 counts)
    def incr_cat_count(self, F2_node):
        self.fam_catcount[F2_node] += 1  # increment the given F2 node counter

    # count the number of F2 nodes with non-zero category count values
    def count_used_cats(self):
        # num_categories = len(self.fam_weights)
        num_nonzero_cats = sum(x > 0 for x in self.fam_catcount)
        return num_nonzero_cats

    # ARTMAP learn cycle
    #
    # Description:
    #   It executes a single ARTMAP learning cycle for a given ARTa baseline vigilance level,
    #   and returns the search history as well as the output prediction for the current input
    #   This prediction may not be the class for the ultimate F2 winning node
    #   as match tracking may have forced the network to search for a different F2 node.
    #   (The predicting F2 node can be found by the last element of the first ARTa search history)
    #
    # Input:
    #  - Fuzzy ARTMAP input vector
    #  - ARTa baseline vigilance level
    #  - target class
    # Output:
    #  - ARTMAP search history
    #  - ARTMAP prediction (after first ARTa recall)
    def learn(self, fam_input, ARTa_baseline_vigilance, fam_target_class):
        #metrics
        #======================

        acc_metric = Accuracy()
        target = torch.tensor([fam_target_class])
        #======================
        count_match_tracking = 0
        fam_search_history = []
        ART_vigilance = ARTa_baseline_vigilance

        # iterate over match tracking cycles, but only until all existing class mappings have been exhausted
        while count_match_tracking < len(self.fam_classes):

            (ART_search_history, winner_match_level) = self.recall(fam_input, ART_vigilance)  # ??

            # append ART search history to ARTMAP search history (index will determine the match tracking cycle)
            fam_search_history.append(ART_search_history)

            # ARTMAP current prediction can be obtained by reading out the class for the winning F2 node (the last in the search history)
            F2_winner = ART_search_history[-1]
            prediction = self.fam_classes[F2_winner]
            if count_match_tracking == 0:  # if this is the first ARTMAP prediction cycle (ie no match tracking yet)
                fam_prediction = prediction  # ARTMAP prediction is this prediction value (which can be 'no prediction' as well)

            # check prediction, if any, against target, and decide on action
            if prediction == FAM_NOCLASS_VALUE:  ## no prediction, uninitialized mapping
                self.fam_classes[F2_winner] = fam_target_class  # simply set the mapping for the target class
                break
            elif prediction == fam_target_class:  ## if correct prediction, done, no need to search for a better F2 node
                break
            else:  # prediction was wrong, need to invoke match tracking
                ART_vigilance = winner_match_level + FAM_MT_VIGILANCE_INCREASE  # raise ARTa vigilance to trigger match tracking
                count_match_tracking += 1

        else:  # all class mappings have been exhausted, something is very wrong here
            print("*** No more ARTMAP classes! ***")

        # Learning step for ARTa once winning F2 node found, and target class prediction confirmed or set (in case of new F2 node)
        # Adjust weight vector of winning F2 node
        F0_AND_F2winner = [min(fam_input[i], self.fam_weights[F2_winner][i]) for i in range(len(fam_input))]
        op1_beta = [i * self.fam_learning_rate for i in F0_AND_F2winner]
        op2_one_minus_beta = [i * (1 - self.fam_learning_rate) for i in self.fam_weights[F2_winner]]
        self.fam_weights[F2_winner] = [sum(x) for x in zip(op1_beta, op2_one_minus_beta)]
        if F2_winner == len(self.fam_weights) - 1:  # if winner is the last F2 node, which was unused before ...
            self.add_F2_node()  # ... add a new F2 node before returning (so there is always one unused F2 node)

        self.incr_cat_count(F2_winner)  # increment the category counter!
        return (fam_search_history, fam_prediction, fam_target_class)

        # ARTMAP recall cycle

    #
    # Description:
    #   It executes a single ART recall cycle with the given vigilance level,
    #   and returns the search history as well as the match level for the winning F2 node
    #   in case match tracking needs to ensue due to wrong prediction
    #   (Prediction can be obtained by checking the class for the winning F2 node, which is the last in the search history)
    #
    # Input:
    #  - Fuzzy ARTMAP input vector
    #  - ARTa vigilance level ()
    #  - target class
    # Output:
    #  - ARTMAP search history
    #  - class prediction
    def recall(self, fam_input, ARTa_vigilance):
        ART_search_history = []
        num_F2_nodes = len(self.fam_weights)
        len_F0 = len(fam_input)
        total_activity_F0 = sum(fam_input)
        # initial match values for each F2 node for the current input presentation
        match_values = [0] * num_F2_nodes
        # initial F2 layer output values (y)
        F2_outputs_y = [0] * num_F2_nodes

        # calculate choice and match values for each F2 node for the current input
        for yi in range(num_F2_nodes):
            size_F0_AND_w = 0
            size_w = 0
            for f0i in range(len_F0):
                size_F0_AND_w += min(fam_input[f0i], self.fam_weights[yi][f0i])
                size_w += self.fam_weights[yi][f0i]
            F2_outputs_y[yi] = size_F0_AND_w / (self.fam_choice_param + size_w)  ## choice
            match_values[yi] = size_F0_AND_w / total_activity_F0

        # execute ART search cycle
        count_f2_search = 0
        while count_f2_search < num_F2_nodes:
            F2_winner = np.argmax(F2_outputs_y)  ## select bottom-up winner
            ART_search_history.append(F2_winner)
            if match_values[F2_winner] < ARTa_vigilance:  ## if winner is not good enough match
                F2_outputs_y[F2_winner] = -1  ## disallow winner from further competition in this cycle
                count_f2_search += 1
                continue
            else:
                break
        else:  # no existing F2 nodes found
            pass #sprint("*** No existing F2 node found! ***")

        # self.fam_catcount[F2_winner] += 1                       # a simple recall will not change the category counter!
        return (ART_search_history, match_values[F2_winner])

# ====================================
from avalanche.evaluation.metrics import Accuracy, BWT, Forgetting, ConfusionMatrix

class ARTIR(Fuzzy_ARTMAP):

    # class variables
    # artir_nfat = ARTIR_NFAT_DEFAULT_VALUE           # default New Feature Activation Threshold value

    # class constructor
    def __init__(self):
        super().__init__()
        self.fmasks = []  # ARTIR internal feature masks
        self.newftoF2 = []  # which F2 node each new internal feature points to
        self.kernels = []  # the list of feature kernels
        self.nfat = ''  # NFAT: New Feautre Activation Threshold (as % of maximum possible activation for the given mask)
        self.fmethod = ''  # the ARTIR feature method ('Basic', 'Kernel', ...)

    # Method: initialize ARTIR data structures
    #
    # Description:
    #  Creates a single new feature mask, which is not used, but will be the first candidate feature mask
    #  The new mask will not link to any F2 node initially
    def initialize(self, dim_input):
        super().initialize(dim_input)
        self.fmasks = [[ARTIR_NEW_FEATURE_MASK_INIT_VALUE for i in range(dim_input)]]  # Initial new feature mask
        self.newftoF2 = [ARTIR_NF_NOF2]  # initially, the new feature will not link to any F2 node
        self.nfat = ARTIR_NFAT_DEFAULT_VALUE
        self.kernels = ARTIR_KERNELS_DEFAULT_LIST
        self.fmethod = ARTIR_FEATURE_METHOD_BASIC

    def new_instance(self, X, num_new_features, artir_nfat, Kernel=False):
        k=X
        size_of_input = len(X)
        artir_1 = ARTIR()
        cat_stats = [0 for i in range(num_new_features + 1)]
        fkernel_list = [[0, 1], [1, 0]]
        if Kernel:
            feature_method = ARTIR_FEATURE_METHOD_KERNEL
            ARTIR.set_kernels(artir_1, fkernel_list)
        else:
            feature_method = ARTIR_FEATURE_METHOD_BASIC

        ARTIR.initialize(artir_1, size_of_input * 2)
        ARTIR.reset_all_cat_counts(artir_1)
        ARTIR.set_nfat(artir_1, artir_nfat)
        ARTIR.set_fmethod(artir_1, feature_method)
        return artir_1, cat_stats, fkernel_list





    # set ARTIR New Feature Activation Threshold parameter (0 <= NFAT <= 1)
    def set_nfat(self, param_nfat):
        self.nfat = param_nfat

    # set ARTIR feature kernel list
    def set_kernels(self, param_kernels):
        self.kernels = param_kernels

    # set ARTIR feature method
    def set_fmethod(self, param_fmethod):
        self.fmethod = param_fmethod



        # Method: ARTIR learn cycle
        #
        # Input:
        #  - current input (assumed to be in complement coding)
        #  - Fuzzy ARTMAP baseline vigilance
        #  - current target class
        # Output:
        #  - whatever is returned by the parent class Fuzzy ARTMAP 'learn' method

    def learn(self, current_input, fam_baseline_vigilance, current_target):
        # expand current input with internal features before Fuzzy ARTMAP training cycle
        new_features = self.calc_features(current_input)  # use object instance variable nfat
        fam_input = current_input + new_features
        return super().learn(fam_input, fam_baseline_vigilance, current_target)

    # Method: Calculate ARTIR new feature outputs based on input
    #
    # Input:
    #  - input vector (assumed to be originating from complement coded inputs)
    # Output:
    #  - list of new feature output values (in elementwise complement coding)

    # USES: Calc_feature_value_generic

    def calc_features(self, input_cc):
        new_features = []
        current_feature_input = input_cc[:]  # this list will collect all ARTIR feature outputs
        for i in range(len(self.fmasks) - 1):  # always ignore the last mask as that is an unitialized one
            new_feature = self.calc_feature_value_generic(current_feature_input, self.fmasks[i])
            # expand the ARTIR fature vector accordingly (in elementwise complement coding)
            new_features.append(new_feature)
            new_features.append(1 - new_feature)
            # expand current feature input (in elementwise complement coding) so it will be the input for the next
            # iteration
            current_feature_input.append(new_feature)
            current_feature_input.append(1 - new_feature)
        return new_features

    # Method: Add a new internal feature to ARTIR
    #
    # Input:
    #  - new feature mask to add
    # Output:
    #  - <no return value>
    #
    # Outline:
    #   This is a simplified procedure that follows the pruning operation after internal search
    #   No direct association of new feature with an F2 node
    #   Expand dimensionality of each F2 weight vector by 2 (new feature in complement coding), and
    #       set the weights of the existing F2 nodes to zero (no connection between new feature and all other F2 nodes)
    #   Create new candidate feature mask with no F2 node to link to
    def add_new_feature(self, new_feature_mask):
        no_of_new_features = len(
            self.fmasks) - 1  # the last one is really an unused one dedicated to candidate features
        size_F2 = len(self.fam_weights)

        # add new feature mask to ARTIR feature masks, essentially replacing the so far unused mask
        self.fmasks[no_of_new_features] = new_feature_mask[:]
        self.newftoF2[no_of_new_features] = ARTIR_NF_NOF2  # new feature does not point to any F2 node

        # add a new unused feature mask, longer than the previous one by 2
        new_unused_fm = [ARTIR_NEW_FEATURE_MASK_INIT_VALUE for _ in range(len(new_feature_mask) + 2)]
        self.fmasks.append(new_unused_fm)
        self.newftoF2.append(ARTIR_NF_NOF2)  # initially, no F2 nodes to link to

        # Changes to ARTMAP: extend each F2 node with two input dimensions, and set the corresponding weights to 0
        # the pruned F2 nodes: 0 because no longer needed the unpruned F2 nodes: 0 because they are not correlated
        # with the output class hence should not have any say in the class mapping for that F2 node
        for i in range(size_F2):
            self.fam_weights[i].append(0)
            self.fam_weights[i].append(0)

        # but the unused F2 node should connect to this new feature, too
        self.fam_weights[-1][-1] = 1
        self.fam_weights[-1][-2] = 1


    # Method: Search new ARTIR internal feature
    #
    # Input:
    #  - maximum number of search cycles
    #  - minimum value for the absolute correlation value (negative or positive)
    # Output:
    #  - New internal feature mask to build into ARTIR
    #  - New feature correlation, NFC (as returned by 'calc_feature_correlation')
    #  - Number of search cycles that was run during search
    #
    # Outline:
    #  Does the following until it either reaches the search cycle limit or the min. expected (asbsolute) correlation threshold
    #    Generate new random feature mask
    #    Calculate the correlation of this new feature with the learned classes (NFC, New Feature Correlation)
    def search_internal_feature(self, max_search_cycles, min_abs_correlation):
        len_fmask = len(
            self.fmasks[-1])  # length of candidate feature is that of the last feature mask (which is currently unused)
        candidate_fmask = [0 for _ in range(len_fmask)]
        search_cycle = 0
        max_abs_cor = 0
        nfc = 0
        while search_cycle < max_search_cycles and max_abs_cor < min_abs_correlation:  # while not reached search
            # cycle limit and max
            candidate_fmask = self.gen_feature_mask_cc(len_fmask, 1, len_fmask / 2 - 1)
            # min. 1 bit pair masked off, min. 1 feature valid (not masked off)
            # candidate_fmask = self.gen_feature_mask_cc(len_fmask, 0, 0)
            (nfc, related_class) = self.calc_feature_correlation_n_categories(candidate_fmask)
            if abs(nfc) > max_abs_cor:
                max_abs_cor = abs(nfc)
            search_cycle += 1
        # else:
        #     if search_cycle == max_search_cycles:
        #         return 0
        if max_abs_cor >= min_abs_correlation:
            self.prune_fmask_categories(candidate_fmask, nfc, related_class)
        return candidate_fmask, nfc, search_cycle

    # Method: ARTIR recall cycle
    #
    # def recall(self, current_input, ARTa_vigilance):
    #     return super().recall(current_input, ARTa_vigilance)

    # Method: Calculate correlation
    #
    # Input:
    #  - feature mask
    # Output:
    #  - New feature correlation (see below)
    #
    # Outline:
    #  Calculates new feature On and Off responses for each F2 class into buckets
    #  Calculates TP and TN for the 0th (first) class
    #  Adds up TP and TN
    #  New Feature Correlation = (TP+TN)-1
    def calc_feature_correlation(self, feature_mask):
        num_classes = 2  # we're only dealing with binary decision problems for now
        # Separate counter buckets for On and Off feture values; each bucket counts per each class
        fon_counter = [0 for i in range(num_classes)]  # On counter bucket
        foff_counter = [0 for i in range(num_classes)]  # Off counter bucket

        # go through each F2 protoype vector
        for i in range(len(self.fam_weights)):
            if self.fam_classes[i] != FAM_NOCLASS_VALUE:  # ignore F2 nodes that do not have associated class
                F2_prototype = self.fam_weights[i]
                feature_output = self.calc_feature_value_generic(F2_prototype,
                                                                 feature_mask)  # response of new feature mask for the current F2 weight vector
                if feature_output == 1:
                    fon_counter[self.fam_classes[i]] = fon_counter[
                                                           self.fam_classes[i]] + 1  # On bucket for the given class
                else:
                    foff_counter[self.fam_classes[i]] = foff_counter[
                                                            self.fam_classes[i]] + 1  # Off bucket for the given class
        # What follows only applies to binary (two-class) problems!
        # Calculate True Positive and True Negative probabilties for the given new feature (with mask and NFAT) and 0th class
        if fon_counter[0] == 0:  # to avoid division by zero
            true_positive_0 = 0
        else:
            true_positive_0 = fon_counter[0] / (fon_counter[0] + foff_counter[0])
      #  print(true_positive_0)

        if foff_counter[1] == 0:  # to avoid division by zero
            true_negative_1 = 0
        else:
            true_negative_1 = foff_counter[1] / (fon_counter[1] + foff_counter[1])
       # print(true_positive_1)

        tp_tn = true_positive_0 + true_negative_1  # the further away from 1.0, the better
        nfc = tp_tn - 1  # new feature correlation: -1 <= ... <=1, zero is no correlation
        return nfc

    def calc_feature_correlation_n_categories(self, feature_mask):
        num_classes = len(np.unique(self.fam_classes)) - 1  # except NO_CLASS
        fon_counter = [0 for _ in range(num_classes)]  # On counter bucket
        foff_counter = [0 for _ in range(num_classes)]  # Off counter bucket
        nfcs = []

        # go through each F2 protoype vector
        for i in range(len(self.fam_weights)):
            if self.fam_classes[i] != FAM_NOCLASS_VALUE:  # ignore F2 nodes that do not have associated class
                F2_prototype = self.fam_weights[i]
                # response of new feature mask for the current F2 weight vector:
                feature_output = self.calc_feature_value_generic(F2_prototype, feature_mask)
                if feature_output == 1:
                    # On bucket for the given class:
                    fon_counter[self.fam_classes[i]] += 1
                else:
                    # Off bucket for the given class
                    foff_counter[self.fam_classes[i]] += 1
        # Calculate True Positive and True Negative probabilties for the new feature (with mask and NFAT) and 0th class
        for i in range(num_classes):
            # The fraction
            true_positive = 0 if fon_counter[i] == 0 else fon_counter[i] / (fon_counter[i] + foff_counter[i])

            numerator = np.sum(foff_counter) - foff_counter[i]
            denominator = numerator + np.sum(fon_counter) - fon_counter[i]
            true_negative = 0 if numerator == 0 else numerator / denominator
            nfcs.append(true_positive + true_negative - 1)  # NFC of the i-th category
        nfc = max(nfcs, key=abs)
        related_class = nfcs.index(nfc)
        return nfc, related_class


    def calc_feature_correlation_n_categories_1(self, feature_mask, num_classes):
        fon_counter = [0 for i in range(num_classes)]  # On counter bucket
        foff_counter = [0 for i in range(num_classes)]  # Off counter bucket
        nfcs = []

        # go through each F2 protoype vector
        for i in range(len(self.fam_weights)):
            if self.fam_classes[i] != FAM_NOCLASS_VALUE:  # ignore F2 nodes that do not have associated class
                F2_prototype = self.fam_weights[i]
                feature_output = self.calc_feature_value_generic(F2_prototype,
                                                                 feature_mask)  # response of new feature mask for the current F2 weight vector
                if feature_output == 1:
                    fon_counter[self.fam_classes[i]] = fon_counter[
                                                           self.fam_classes[i]] + 1  # On bucket for the given class
                else:
                    foff_counter[self.fam_classes[i]] = foff_counter[
                                                            self.fam_classes[i]] + 1  # Off bucket for the given class
        # What follows only applies to binary (two-class) problems!
        # Calculate True Positive and True Negative probabilties for the given new feature (with mask and NFAT) and 0th class


        for l in range(num_classes):
            Tn = [] # True negative parts vector, multiply all the indexes with each other (product)
            if fon_counter[l] == 0:  # to avoid division by zero
                true_positive = 0
            else:
                true_positive = fon_counter[l] / (fon_counter[l] + foff_counter[l]) #Same true positive calculating as in the binary case

            for k in range(num_classes):
                if k != l:
                    if foff_counter[k] == 0: # to avoid division by zero
                        negative = 0 # This is a "part" of true negative
                    else:
                        negative = foff_counter[k] / (fon_counter[k] + foff_counter[k]) # This is a "part" of true negative

                    Tn.append(negative)

            true_negative = np.prod(Tn) #Multiply the content of Tn[]
            nfcs.append(true_positive + true_negative - 1) # NFC of the l-th category
        nfc = np.max(np.abs(nfcs))# after we have the absolute value, select the maximum
        return nfc

    def calc_feature_correlation_n_category(self, feature_mask, num_classes):
        bucket_matrix = [[0 for l in range(num_classes)] for i in range(3)]
        #fon_counter = [0 for i in range(num_classes)]  # On counter bucket
        #foff_counter = [0 for i in range(num_classes)]  # Off counter bucket

        # go through each F2 protoype vector
        for i in range(len(self.fam_weights)):
            if self.fam_classes[i] != FAM_NOCLASS_VALUE:  # ignore F2 nodes that do not have associated class
                F2_prototype = self.fam_weights[i]
               # print(i)
                feature_output = self.calc_feature_value_generic(F2_prototype,feature_mask)
                # response of new feature mask for the current F2 weight vector
                if feature_output == 1:
                    bucket_matrix[0][self.fam_classes[i]] = \
                        bucket_matrix[0][self.fam_classes[i]] + 1 # On bucket part for the given class

                else:
                    bucket_matrix[1][self.fam_classes[i]] =\
                        bucket_matrix[1][self.fam_classes[i]] + 1 # Off bucket part for the given class


        probability_vector = [0 for i in range(num_classes)]
        # Vector to store the probabilities by column
        category_minimum = sum(bucket_matrix[2])/num_classes
        # A threshold value to check if there is enough prototype vectors to make a new feature

        for b in range(len(bucket_matrix[2])):
            bucket_matrix[2][b] = bucket_matrix[0][b] + bucket_matrix[1][b]
            #sum of the prototype vectors per column


        for i in range(len(probability_vector)):
            if bucket_matrix[0][i]==0:
                on_p = 0
            else:
                on_p = \
                   abs( bucket_matrix[0][i]/(bucket_matrix[0][i] + bucket_matrix[1][i]) -1)
            #positive correlation
            if bucket_matrix[1][i] == 0:
                off_p = 0
            else:
                off_p =\
                   abs( bucket_matrix[1][i]/(bucket_matrix[0][i] + bucket_matrix[1][i]) -1)
            #negative correlation

            if on_p >= off_p: #check which one should be used
                probability_vector[i] = on_p
            else:
                probability_vector[i] = off_p

        for i in range(len(probability_vector)):
            max_index = probability_vector.index(max(probability_vector))  #check which one has the biggest correlation
            if bucket_matrix[2][max_index] > category_minimum: # if it reaches threshold of sum
                return probability_vector[max_index] #new feature correlation
            else:
                probability_vector[max_index] = 0 #never get value again

        return 0



    # Method: Prune ARTIR categories
    #
    # Input:
    #  - new candidate feature mask
    #  - New Feature Correlation (NFC), only its sign is used to determine which nodes to prune
    # Output:
    #  - (ARTIR data structure with pruned weights and classes)
    #
    # Outline:
    #
    #  Resets all F2 node weights and their class mappings of all F2 nodes whose classes correlate with the new candidate feature mask
    #  Those nodes that do not correlated with the feature mask directly are left intact
    #  Typically called after 'search_internal_features'
    def prune_fmask_categories(self, feature_mask, nfc, related_class):
        num_categories = len(self.fam_weights)
        i = 0
        while i < num_categories:
            f_value = self.calc_feature_value_generic(self.fam_weights[i], feature_mask)
            if nfc > 0:  # if NFC > 0, positive correlation with the respective class
                if f_value == 1 and self.fam_classes[i] == related_class:  # if feature correctly predicts related class
                    super().reset_F2_node(i)
                    num_categories -= 1
                    i-=1
                elif f_value == 0 and self.fam_classes[i] != related_class:  # if feature correctly predicts other class
                    super().reset_F2_node(i)
                    num_categories -= 1
                    i-=1
            elif nfc < 0:  # if NFC < 0, negative correlation with the respective class
                if f_value == 1 and self.fam_classes[i] != related_class:  # if feature correctly predicts class 1
                    super().reset_F2_node(i)
                    num_categories -= 1
                    i-=1
                elif f_value == 0 and self.fam_classes[i] == related_class:  # if feature correctly predicts class 0
                    super().reset_F2_node(i)
                    num_categories -= 1
                    i-=1
            # else:
            i += 1  # if NFC = 0, no correlating feature, no pruning
    # Method: Calculate feature output (generic method)
    #
    # Input:
    #  - input vector
    #  - feature_mask
    # Output:
    #  - feature output
    #
    # Description:
    #  Generic method to calculate an internal feature value
    #  It calls specific methods depending on the object variable 'fmethod' that selects the method
    def calc_feature_value_generic(self, input_vector, feature_mask):
        if self.fmethod == ARTIR_FEATURE_METHOD_BASIC:
            return ARTIR.calc_feature_value_basic(input_vector, feature_mask, self.nfat)
        elif self.fmethod == ARTIR_FEATURE_METHOD_KERNEL:
            return ARTIR.calc_feature_value_kernel(input_vector, feature_mask, self.kernels)
        else:
            return -1

    # Method: Calculate feature output using a simple mask (the original idea)
    #
    # Input:
    #  - input vector
    #  - feature mask
    #  - NFAT
    # Output:
    #  - feature output
    @staticmethod
    def calc_feature_value_basic(input_vector, feature_mask, feature_threshold):
        size_feature_mask = sum(feature_mask)  # total size of feature mask (needed for thresholding on %)
        total_vector_match = 0
        for i in range(len(input_vector)):
            total_vector_match += min(input_vector[i], feature_mask[i])
        if total_vector_match / size_feature_mask >= feature_threshold:  # total match with feature exceeds threhsold (as % of mask size)?
            feature_value = 1
        else:
            feature_value = 0
        return feature_value

    # Method: Calculate feature output with given feature kernel
    #
    # Input:
    #  - input vector (complement coded)
    #  - feature mask (complement coded, but really only the masking off bit pair should be (0,0))
    #  - feature kernel list (see format below)
    # Output:
    #  - feature output (0 or 1)
    #
    # Description:
    #  Checks if given feature kernel matches the input at ANY location
    #  Ignores positions where feature mask is (0,0) in complement coding (otherwise mask pairs can be of any other combination)
    #  Assumes input in complement coding (mask enough to have at least one bit in the CC pair to be '1' to count)
    #  Kernel check wraps around so start and end of input vector are considered neighbors
    #  '0' and '1' map to '01' and '10' in complement coding
    #  Kernel list format: [K1, K2, ..., KN] where Ki = [Ki1, Ki2, ..., KiM], NOT in complement coding
    #   If axíny one of the kernels matches the non-masked input vector in any position, the output will be '1'. Matching is checked in wrap around mode (modulo length of the input vector)
    #  Currently no checking of kernel vs. mask, so the same input bit can be checked more than once due to wrap around
    @staticmethod
    def calc_feature_value_kernel(input_vector, feature_mask, feature_kernel_list):
        for i in range(0, len(input_vector), 2):  # loop through each CC bit pair of input vector
            for kli in range(len(feature_kernel_list)):  # loop through each kernel in the list
                fkernel_cc = cautils.ca_complement_code_elements(feature_kernel_list[kli])
                # convert current kernel into complement coded vector
                fmi = i  # starting element of the input vector for checking matching
                num_kmatch = 0
                for k in range(0, len(fkernel_cc), 2):  # loop through the CC version of the kernel vector
                    # skip bit pairs where mask is (0,0), and wrap around (mod len(input_vector))
                    while feature_mask[fmi] == 0 and feature_mask[fmi + 1] == 0:
                        fmi = (fmi + 2) % len(input_vector)
                    if input_vector[fmi] != fkernel_cc[k]:  # enough to check first bit in complement coded pair
                        break  # no match with kernel, break from loop
                    else:
                        fmi = (fmi + 2) % len(
                            input_vector)  # if match, check next bit in kernel and input (wrap around)
                        num_kmatch += 1  # count the number of matching bits in kernel
                        continue
                if num_kmatch == len(feature_kernel_list[kli]):  # if all bits in current kernel matched, return TRUE
                    feature_value = 1
                    return feature_value
        # if all kernels have been checked in all positions in input vector
        feature_value = 0
        return feature_value

    # Method: Generate feature mask - generic version
    #
    # Input:
    #  - length of mask (it will be in complement coding, so input should be an even number), it should also be greater than 2
    #  - min. number of (0,0) pairs in the generated mask
    #  - max. number of (0,0) pairs in the generated mask
    # Output:
    #  - feature mask in complement coding (mask size cannot be zero)
    #
    # Description:
    #  Generates a sequence of random 0's and 1's in elementwise complement coding
    #  The number of (0,0) pairs, i.e. the masked off bits, must be between the min. and max. zeros (input parameters)
    @staticmethod
    def gen_feature_mask_cc(mask_length, min_zeros, max_zeros):
        if mask_length % 2 != 0:  # check if input vector length is an even number
            print("ERROR: input vector length is not an even number!")
            return
        elif mask_length <= 2:  # mask cannot be a single bit (or length of 2 in complement coding)
            print("ERROR: mask length must be greater than 2!")
            return
        elif min_zeros > max_zeros:
            print("ERROR: Min. zero bits cannot be greater than max. zero bits!")
            return
        elif max_zeros * 2 >= mask_length:
            print("ERROR: Mask cannot be all zeros!")
            return

        mask = [0 for i in range(mask_length)]  # set all mask bits to zero initially

        for i in range(0, mask_length, 2):  # generate random mask with no zero bit pairs first
            maskbit = random.randint(0, 1)
            mask[i] = maskbit
            mask[i + 1] = 1 - maskbit

        num_zeros = random.randint(min_zeros, max_zeros)  # random number of zero pairs in between min. and max.

        zero_pos = [i for i in range(0, mask_length,
                                     2)]  # the list of indices from which the position of the (0,0) pair can be drawn at random

        for i in range(num_zeros):
            randpos = random.randint(0, len(zero_pos) - 1)  # select a random element from the list of indices
            zeroindex = zero_pos.pop(randpos)  # remove that element from the list and get its value
            # set the bit pair in CC to (0,0)
            mask[zeroindex] = 0
            mask[zeroindex + 1] = 0

        return mask

    # Method: Generate fature mask -- SIMPLE VERSION (the first one before making it more general)
    #
    # Input:
    #  - length of mask (it will be in complement coding, so input should be an even number), it should also be greater than 2
    # Output:
    #  - feature mask in complement coding (mask size cannot be zero)
    #
    # Description:
    #  Generates a sequence of random 0's and 1's in elementwise complement coding
    #  Mask cannot be all 0's, and must contain at least one (0, 0) pairs (i.e. masked out bit)
    @staticmethod
    def gen_feature_mask_cc_simple(mask_length):
        if mask_length % 2 != 0:  # check if input vector length is an even number
            print("ERROR: input vector length is not an even number!")
            return
        elif mask_length <= 2:  # mask cannot be a single bit (or length of 2 in complement coding)
            print("ERROR: mask length must be greater than 2!")
            return
        mask = [0 for i in range(mask_length)]  # set all mask bits to zero initially
        while sum(mask) == 0:  # make sure mask will have at least one non-zero bit
            for i in range(0, mask_length, 2):  # loop through mask in steps of 2 (due to complement coding)
                maskbit = random.randint(-1, 1)  # maskbit = -1, 0, +1
                if maskbit == 1:
                    mask[i] = 1
                    mask[i + 1] = 0
                elif maskbit == -1:
                    mask[i] = 0
                    mask[i + 1] = 1
                # if maskbit = 0, do nothing, this bit will be (0,0) in complement coding

        if sum(mask) == mask_length / 2:  # make sure there is at least one bit (in CC) masked out
            zeroindex = random.randint(0, mask_length / 2 - 1)
            mask[zeroindex * 2] = 0
            mask[zeroindex * 2 + 1] = 0

        return mask

# ====================================


