from bbox_gen.utils.box_processing import *
from bbox_gen.utils.ocr_processing import *
from bbox_gen.utils.text_processing import *
import numpy as np
import pandas as pd
from PIL import Image
import os
import json
import time, math
def getTextMatchCheckers(k, max_, max_len_):
    assert max_ % 2 != 0, "Make sure the maximum boxes is odd number"
    sides = round(max_ % 2)
    vals = []
    for i in range(0,sides+1):
        vals.append(k-(i+1))
        vals.append(k+(i+1))
    vals.append(k)
    final_vals = [val for val in vals if val >= 0 and val < max_len_]

    return sorted(final_vals)


#Given the original sentence, return the order of texts
def getTextOrder(text, max_=5):
    textSplit = text.split(' ')
    textOrderDict = {k:{'original':v, 'matched':False, 'pred_matchers':getTextMatchCheckers(k,max_,len(textSplit))} for k,v in enumerate(textSplit)}

    return textOrderDict