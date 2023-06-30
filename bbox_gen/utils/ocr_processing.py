from bbox_gen.utils.box_processing import *
from bbox_gen.utils.ocr_processing import *
from bbox_gen.utils.text_processing import *
import numpy as np
import pandas as pd
from PIL import Image
import os
import json
from Levenshtein import distance as lev
import time, math


def getLevMatched(predictions, text_order):
    lmatrix = lev_matrix(predictions, text_order)

    predictions, text_order, lmatrix = findPerfectMatches(predictions, text_order, lmatrix)

    predictions, text_order, lmatrix = match121(predictions, text_order, lmatrix)
    predictions, text_order = parsePredictions(predictions, text_order)

    predictions, text_order = parseTxtOrder(predictions, text_order)

    return predictions, text_order

def lev_matrix(predictions, text_order):
    lev_matrix = np.zeros((len(predictions), len(text_order)))

    for pred_key, pred_value in predictions.items():
        pred_text = pred_value['prediction']


        lev_arr = [1e10] * len(text_order)

        for text_key, text_value in text_order.items():
            if pred_key in text_value['pred_matchers']:
                orig_text = text_value['original']

                lev_arr[text_key] = lev(pred_text, orig_text)

        lev_matrix[pred_key, :] = lev_arr

    return lev_matrix


#===================================================================
#Matching
#===================================================================
def findPerfectMatches(predictions, text_order, lmatrix):
    pred_locs, text_locs = np.where(lmatrix == 0)

    for pred_loc, text_loc in list(zip(pred_locs,text_locs)):
        predictions[pred_loc]['original'] = text_order[text_loc]['original']
        predictions[pred_loc]['matched'] = True
        text_order[text_loc]['matched'] = True

    return predictions, text_order, lmatrix

def match121(predictions, text_order, lmatrix):

    #Go through each row of lmatrix which corresponds to the lev distances of the prediction to nearby words in sentence
    for index, pred in enumerate(lmatrix):
        #Set all matched levs to max (1e10) value
        for j, i in text_order.items():
            if i['matched']:
                pred[j] = 1e10

        #If not matched yet
        if not(predictions[index]['matched']):

            #Find the minimum distance word and distance
            min_lev = pred.min()
            min_lev_index = pred.argmin()

            if min_lev >= len(predictions[index]['prediction']):
                predictions[index]['original'] = 'UKN'
                continue

            predictions[index]['original'] = text_order[min_lev_index]['original']

            #If a threshold is met for the lev distance then set matched to true
            if min_lev < abs(len(predictions[index]['prediction']) - len(text_order[min_lev_index]['original'])) or min_lev <= 2:
                predictions[index]['matched'] = True
                text_order[min_lev_index]['matched'] = True

    return predictions, text_order, lmatrix

def getItemKey(dict_, key, value, matched=False):

    for dict_key, dict_value in dict_.items():
        if dict_value[key] == value and dict_value['matched'] == matched:
            return dict_key

    return None

def remove_non_consec(list_):
    if len(list_) <= 1:
        return list_

    list_ = sorted(list_)
    final_list = []
    for index, i in enumerate(list_):
        if index == 0:
            if i == list_[index+1] - 1:
                final_list.append(i)
        elif index == len(list_)-1:
            if i == list_[index-1] + 1:
                final_list.append(i)
        else:
            if i == list_[index-1] +1 and list_[index-1] in final_list:
                final_list.append(i)

    return sorted(final_list)



def parsePredictions(predictions, text_order):

    for pred_key, pred_value in predictions.items():
        if not(pred_value['matched']):
            orig_match = pred_value['original']
            text_key = getItemKey(text_order, 'original', orig_match, matched=False)
            if text_key != None:
                pos_matched = text_order[text_key]['pred_matchers']

                false_original_key = [k for k in pos_matched if not(text_order[k]['matched'])]
                false_original_key = remove_non_consec(false_original_key)


                match_3 = [j for j in [false_original_key[i:i+3] for i, _ in enumerate(false_original_key)] if len(j) == 3]
                match_2 = [j for j in [false_original_key[i:i+2] for i, _ in enumerate(false_original_key)] if len(j) == 2]

                all_matches = []
                all_matches.extend(match_3)
                all_matches.extend(match_2)

                final_text = ""
                final_min_lev = 1000
                final_match = []

                for match in all_matches:
                    text = "".join([text_order[k]['original'] for k in match])
                    text_to_add = " ".join([text_order[k]['original'] for k in match])
                    min_lev = lev(text, pred_value['prediction'])

                    if min_lev < len(text) or min_lev < 4:
                        if final_min_lev > min_lev:
                            final_text = text_to_add
                            final_min_lev = min_lev
                            final_match = match


                pred_value['original'] = final_text
                pred_value['matched'] = True

                for i in final_match:
                    text_order[i]['matched'] = True
            else:
                continue

    return predictions, text_order

def parseTxtOrder(predictions, text_order):

    final_dict = {}
    count = 0
    skip = 0

    for text_key, text_value in text_order.items():
        if skip > 0:
            skip -= 1
            continue

        if text_value['matched']:
            orig_text = text_value['original']
            keys = [k for k,v in predictions.items() if orig_text in v['original'] and v['matched'] and not(v in final_dict.values())]
            if len(keys) > 0:
                final_dict[count] = predictions[keys[0]]
                count+= 1
                if len(predictions[keys[0]]['original'].split(" ")) > 1:
                    skip = len(predictions[keys[0]]['original'].split(" ")) - 1
        else:
            final_dict[count] = {
                'prediction':"UKN",
                'probability': -1,
                'box':[],
                'matched':True,
                'sent_order':-1,
                'original': text_value['original']
            }
            count += 1

    return final_dict, text_order



