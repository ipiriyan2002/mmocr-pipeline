from bbox_gen.utils.box_processing import *
from bbox_gen.utils.ocr_processing import *
from bbox_gen.utils.text_processing import *
import numpy as np
import pandas as pd
from PIL import Image
import os
import json
import torch.utils as tu
from Levenshtein import distance as lev
import time, math

#===================================================================
#Preprocssing
#===================================================================

"""
Calculate the average heights of given boxes
if the height of a particular box is less than minimum height, then box is not used to calculate average height
"""
def calcAverageHeights(boxes, min_height=5):

    heights = [((box[3]-box[1])+1) for box in boxes if ((box[3]-box[1])+1) >= min_height]

    avg_height = sum(heights) / len(heights)
    return avg_height

"""
Remove predicted boxes with a less than average height
"""
def removeBelowAverageHeight(preds, thresh_pad=20, min_height=5, min_width=2):
    new_preds = {}
    count = 0

    #All boxes, used to calculate average height
    boxes = [out['box'] for out in preds.values()]
    averageHeight = calcAverageHeights(boxes)
    print(f"avg height: {averageHeight}") #Printing average height for debugging purposes

    for value in preds.values():

        #calculate height and width of predicted box
        height = (value['box'][3] - value['box'][1]) + 1
        width = (value['box'][2] - value['box'][0]) + 1

        #Using a pad value to add a error rate to average height such that smaller and bigger valid boxes are taken into account
        #calculate a minimum average height, wherein, the lowest value is minimum height
        min_avg_height = min_height if (averageHeight-thresh_pad) < min_height else averageHeight-thresh_pad
        #if the height of the box is above minimum average height and below a max average height
        #and the width is above valid minimum width
        #Accept prediction as valid
        if (height >= min_avg_height) and (height <= averageHeight+(thresh_pad*1.5)) and (width > min_width):
            new_preds[count] = value
            count += 1

    return new_preds


#===================================================================
#Main
#===================================================================
"""
Given the predicted words, 
Using the sentence order, get the text neighbours and sentence neighbours for each specific predicted box
"""
def getNeighbours(paired_pred):
    for pred_key, pred_value in paired_pred.items():
        #Sentence
        sent_ = pred_value['sent_order']

        #All possible neighbours
        pos_neighbours = list(set([nb for nb in [pred_key-2, pred_key-1,pred_key,pred_key+1,pred_key+2] if nb >=0 and nb < len(paired_pred)]))
        #Possible neighbouring sentences
        pos_nb_sents = {k:paired_pred[k]['sent_order'] for k in pos_neighbours}
        #Get the sentence neighbours
        sent_neighbours = [k for k,v in pos_nb_sents.items() if v == sent_]
        #Store the neighbours
        pred_value['text_neighbours'] = list(pos_nb_sents.keys())
        pred_value['sent_neighbours'] = sent_neighbours

    return paired_pred
"""
Given unordered prediction of pairs, return ordered predictions
"""
def orderPaired(paired):

    #Get the predicted boxes
    boxes = [item['box'] for item in paired.values()]
    #Get the order of sentences
    sentOrder = getSentenceOrder(boxes)

    boxOrd = {}
    initbox = 0

    for sentKey, sent in sentOrder.items():
        #get numpy array version of boxes in given sentence
        sent = np.array(sent)
        #Order boxes, according to x-center coordinate
        ordS = sent[sent[...,0].argsort()]

        #Unpack the boxes wrt to sentences
        for os in ordS:
            boxOrd[initbox] = {'box':os, 'sent_order': sentKey}
            initbox += 1

    ordPairs = {}

    #Assign the order to the paired dict and hence order it
    for sentKey, sentBox in boxOrd.items():
        pairKey = getBoxKey(paired, list(sentBox['box']))
        paired[pairKey]['sent_order'] = sentBox['sent_order']
        ordPairs[sentKey] = paired[pairKey]

    return ordPairs

#===================================================================
#Box Sentence Procesiing
#===================================================================

#Given all the pairs of prediciton and a box, return the key at which predicted box is
def getBoxKey(pairs, box):

    for key, item in pairs.items():
        if item['box'] == box:
            return key

    return None

#Given a box and origin, get the distance between the origin and the box
def calcOrginDistance(box, origin=[0, 0]):
        
    ox, oy = origin

    dist = ((ox - box[0])**2 + (oy - box[1])**2)**0.5

    return dist

#Given a bounding box calculate the center point
def calcCenterPoint(box):
    x1,y1,x3,y3 = box

    cx = x1 + (x3-x1) // 2
    cy = y1 + (y3-y1) // 2

    return [cx,cy]

#Get the average height
def getAverageHeight(boxes):
    heights = [box[3]-box[1] for box in boxes]
    return sum(heights) // len(heights)

#Get the sentence order
def getSentenceOrder(boxes):
    boxes = np.array(boxes)
    #Calculate the center points for all boxes
    box_cen = np.array([calcCenterPoint(box) for box in boxes])
    #Get the order for the box centers
    ord_ = box_cen[...,1].argsort()
    #Order the boxes and the box centers
    boxes = boxes[ord_]
    box_cen = box_cen[ord_]
    #pair them
    paired = list(zip(box_cen, boxes))

    #dictionary of sentence orders
    sentOrd = {}

    #Get the average height
    avg_height = getAverageHeight(boxes) // 2

    prev = -1
    sen = 0

    #Since the boxes are ordered according to height and width, given a list of predicted boxes
    #[n1,n2,n3,n4....nk] -> n1 would be first box whilst nk would be the last box in the image wrt to height and width
    for pair in paired:
        #if no keys exists, use the first
        if len(sentOrd.keys()) == 0:
            sentOrd[sen] = [pair[1]]
            prev = pair[0][1]
            continue

        height = pair[0][1]
        #If the height of a given box is within a certain threshold of the previous height,
        #then it is part of the previous height's sentence
        #If not it is in a new sentence
        if prev <= (height+avg_height) and prev >= (height-avg_height):
            sentOrd[sen].append(pair[1])
        else:
            sen += 1
            sentOrd[sen] = [pair[1]]
            prev = height

    return sentOrd

#===================================================================
#Box Generation
#===================================================================

"""
set ignore to True/False depending on factors such as box available, and or ignore key is in dict
"""
def addIgnore(pairs):

    for value in pairs.values():
        if not('ignore' in value.keys()):
            if len(value['box']) == 4:
                value['ignore'] = False
            else:
                value['ignore'] = True
        else:
            if len(value['box']) != 4:
                value['ignore'] = True
    return pairs


"""
Generate bounding boxes given neighbouring bounding boxes, 

So say we have spacial location :: n1, n2, n3
where n1's and n3's bboxes are known, then using this knowledge, aswell as the original width and height of image,
calculate an abstract interpretation for the n2's bounding box

The bounding box calculating through this method is not 100% correct
We are using a very small pad value to make sure the calculated box fully shows the text inside and does not cover it

Inconsistent bounding boxes rarely rise in the case of no previous and next bounding box, if so, bounding box is not generated
"""
def generateBBox(bbox1, bbox2, orig_width, orig_height, pad=4):

    # get x1,y1 coordinates of next (n3) bounding box as the x2, y2 of current box (n2)
    px2, py2 = bbox2[0], bbox2[1]
    # get x3, y3 coordinates of previous bounding box as the x4, y4 of current box (n2)
    px4, py4 = bbox1[2], bbox1[3]

    #Get possible width and height of current box (n2)
    width = (px2 - px4)
    height = (py4 - py2)

    #Make sure coordinates are withing boundary of image
    x1 = max(0, px4)
    y1 = max(0, py2 - (pad // 2))

    x3 = min(x1 + (width-1), orig_width)
    y3 = min(y1 + (height-1) + (pad // 2), orig_height)

    return [x1,y1,x3,y3]


"""
Given the predicted boxes, calculate a rough estimate for the width of each character (since this is different per image / font size)
"""
def averageCharWidth(preds):
    char_widths = []

    #using original text and predicted bounding box width
    #to get average width
    # a mean value of all prediction is used as avg character width else, a default estimate of 10 is used.
    for pred_key, pred in preds.items():
        if pred['box'] != []:
            pred_text = pred['original']
            pred_box = pred['box']
            pred_width = pred_box[2] - pred_box[0]

            avg_chars = pred_width // len(pred_text)
            char_widths.append(avg_chars)

    try:
        return sum(char_widths) // len(char_widths)
    except:
        return 10

"""
Given the key of unknown box, predictions, original width
get the possible neighbouring boxes and or empty boxes

"""
def getNeigbourBoxes(key, preds, orig_width, orig_height, pad=10):
    #Get original text at key, avg character width and the possible text width at key
    text_ = preds[key]['original']
    avg_char_width = averageCharWidth(preds)
    text_width = len(text_) * avg_char_width

    #Previous box and next box
    prev_box = []
    next_box = []

    #Get the neighbour boxes if they are not empty
    if key-1 >= 0:
        if preds[key-1]['prediction'] != '' and len(preds[key-1]['box']) == 4:
            prev_box = preds[key-1]['box']

    if key+1 < len(preds):
        if preds[key+1]['prediction'] != '' and len(preds[key+1]['box']) == 4:
            next_box = preds[key+1]['box']

    #Empty the boxes if previous box or next box is in a different line
    if key-1 >= 0 and key+1 < len(preds):
        if preds[key+1]['sent_order'] - preds[key-1]['sent_order'] > 0:
            prev_box = []
        elif preds[key+1]['sent_order'] - preds[key-1]['sent_order'] < 0:
            next_box = []


    #Assuming text is in same level as next_box
    if prev_box == [] and next_box != []:
        #Creating a phantom box of width pad, for the previous box if it does not exist
        x1 = max(0, next_box[0] - pad - text_width)
        y1 = next_box[1]

        x3 = max(0, x1 + pad)
        y3 = next_box[3]

        prev_box = [x1,y1,x3,y3]
    #Assuming text is in the same level as prev_box
    elif prev_box != [] and next_box == []:
        #Creating a phantom box for the next box if it does not exist
        x3 = min(orig_width, prev_box[2] + 10 + text_width)
        y3 = prev_box[3]

        box_height = prev_box[3] - prev_box[1]
        x1 = min(orig_width, x3 - pad)
        y1 = prev_box[1]

        next_box = [x1,y1,x3,y3]

    return prev_box, next_box


"""
Generate boxes if no predicted boxes exist and or the prediction is UKN,
ignore is set to True for predicted boxes in case generated bounding box is not perfect

Can be removed if needed when needed
"""
def generateUnknownBoxes(paired_pred, img_meta):
    for out_key, out in paired_pred.items():
        out['ignore'] = False
        if out['prediction'] == 'UKN' and out['box'] == []:
            #get neighbouring boxes
            prev_box, next_box = getNeigbourBoxes(out_key, paired_pred, img_meta['width'], img_meta['height'])
            try:
                #try generating the boxes
                genBox = generateBBox(prev_box, next_box, img_meta['width'], img_meta['height'])
            except:
                #if not set gen box to []
                genBox = []
            out['box'] = genBox
            out['ignore'] = True
    return paired_pred

#===================================================================
#Order Sentence
#===================================================================

def order_sent(pred_boxes):
    values = list(pred_boxes.values())

    #Get seperate sentences
    sent_dict = {}
    for v in values:
        #Seperate prediction wrt to sentence order
        if v['sent_order'] in sent_dict.keys():
            sent_dict[v['sent_order']].append(v)
        else:
            sent_dict[v['sent_order']] = []
            sent_dict[v['sent_order']].append(v)

    #group sentences
    out_dict = {}

    for index, (sent, sent_vals) in enumerate(sent_dict.items()):
        try:
            boxes = [sent_val['box'] for sent_val in sent_vals if len(sent_val['box']) == 4]
            boxes = np.array(boxes)
            #Calculate the center points for all boxes
            box_cen = np.array([calcCenterPoint(box) for box in boxes])
            #Get the order for the box centers wrt to x-center
            ord_ = box_cen[...,0].argsort()
            #Order the boxes and the box centers
            sent_vals = list(np.array(sent_vals)[ord_])


            x_min = sent_vals[0]['box'][0] #First box's x1 value
            y_min = min([box[1] for box in boxes]) #lowest height from all y1s
            x_max = sent_vals[-1]['box'][2] #Last box's x3 value
            y_max = max([box[3] for box in boxes]) #highest height from all y3s

            #Get the bounding box
            joined_box = [x_min,y_min, x_max, y_max]

            #Join the words together to form sentences for both original and prediction
            joined_sent_original = [sent_val['original'] for sent_val in sent_vals if not(sent_val['ignore'])]
            joined_sent_original = " ".join(joined_sent_original)
            joined_sent_pred =  [sent_val['prediction'] for sent_val in sent_vals]
            joined_sent_pred = " ".join(joined_sent_pred)

            #ignore is set to True if only all words in sentence had their ignore set to True
            ignore = all([sent_val['ignore'] for sent_val in sent_vals])

            out_dict[index] = dict(original=joined_sent_original, prediction=joined_sent_pred, box=joined_box, ignore=ignore, sent_order=sent)
        except:
            continue

    return out_dict