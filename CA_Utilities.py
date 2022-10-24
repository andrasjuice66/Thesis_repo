# -*- coding: utf-8 -*-
"""
Utility functions for 'Creative ART' implementation

Author: Gusztav Bartfai
Date: Jan 22, 2021

Description:
    Misc utilities for Creative ART implementation (also needed for Fuzzy ARTMAP)
    Functions defined:
        - generate_random_mask
        x (- generate_random_mask_cc)
        - complement_code_vector
        - complement_code_elements
        x (- ca_feature_mask_value) 

"""
import random

#------------------------------------
# Function: Generate random mask
#
# Input:
#  - size of mask (list of 1s and 0s) to generate
# Output:
#  - mask of given size (all zeros cannot be returned, mask size must be greater than 0)
def ca_generate_random_mask(mask_length):

    mask_size = 0
    while mask_size == 0:
        mask = []
        for i in range(0,mask_length):
            n = random.randint(0,1)
            mask.append(n)
        mask_size = sum(mask)
    return(mask)

#------------------------------------
# Function: Generate random mask and return it in elementwise complement coding
#   i.e., double the length of the original mask
#
# Input:
#  - length of mask (list of 1s and 0s) to generate (without complement coding)
# Output:
#  - mask of given size x2 due to complement coding
#
# THIS IS NOW REPLACED WITH ARTIR STATIC METHOD 'gen_feature_mask_cc'
# BESIDES, THIS METHOD DID NOT GENERATE A TRUE MASK
# def ca_generate_random_mask_cc(mask_length):

#     mask = []
#     for i in range(mask_length):
#         n = random.randint(0,1)
#         mask.append(n)
#     mask_cc = ca_complement_code_elements(mask)
#     return(mask_cc)

#------------------------------------
# Function: Complement code input vector by appending the entire vector in CC
#
# Input:
#  - vector to complement code
# Output:
#  - input vector complement coded
#   format: [i1, i2, ..., in, i1c, 12c, ..., inc]
def ca_complement_code_vector(vector_in):

    vector_in_cc = vector_in[:]
    for i in range(len(vector_in)):
            vector_in_cc.extend([1-vector_in[i]])
    return (vector_in_cc)

#------------------------------------
# Function: Complement code input vector by pairing input elements with their CC
#
# Input:
#  - vector to complement code
# Output:
#  - input vector complement coded
#    format: [i1, i1c, i2, i2c, ..., in, inc]
def ca_complement_code_elements(vector_in):

    vector_in_cc = []
    for i in range(len(vector_in)):
            vector_in_cc.extend([vector_in[i]])
            vector_in_cc.extend([1-vector_in[i]])
    return (vector_in_cc)

#------------------------------------
# Function: Calculate output of a masking feature
#
# Input:
#  - input vector
#  - mask vector
#  - minimum matching level as percentage of mask total size, i.e.
#      NFAT: New Feautre Activation Threshold (as % of maximum possible activation for the given mask)
#
# Assumptions:
#  - input and mask are of the same length
#  - mask vector size is greater than zero
# 
# Output:
#  - output value
#
# THIS IS NOW REPLACED WITH ARTIR STATIC METHOD 'calc_feature_value'
# def ca_feature_mask_value(
#         current_input,
#         feature_mask,
#         new_feature_activation_threshold):
    
#     total_mask_size = sum(feature_mask)
#     total_mask_vector_match = 0
#     for j in range(len(current_input)):
#         total_mask_vector_match += min(current_input[j],feature_mask[j])
#     if total_mask_vector_match/total_mask_size >= new_feature_activation_threshold:
#         feature_output = 1
#     else:
#         feature_output = 0
#     return (feature_output)

#====================================
