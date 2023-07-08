#Custom imports
from bbox_gen.utils.box_processing import *
from bbox_gen.utils.ocr_processing import *
from bbox_gen.utils.text_processing import *
from utils.box_translator_utils import *
#MMOCR imports
from mmocr.apis import MMOCRInferencer

#Others
from PIL import Image
import os
import numpy as np

"""
Box generator class

Uses MMOCR inferences and Lev Distance to generate bounding boxes for words/sentences in image
"""
class MMOCRBoxGenerator:

    def __init__(self, det_model="dbnet", det_weights=None, rec_model="satrn", rec_weights=None,
                 device="cuda:0", save_dir="./box_gen/", max_neighbours=5):

        #Models and Devices
        self.det_model = det_model
        self.rec_model = rec_model
        self.device = device
        self.save_dir = save_dir

        try:
            os.makedirs(save_dir)
        except:
            pass

        #Defines the window size
        # Given a value of 5:
        # n1, n2 , current word, n3, n4 -> Including the current word, the window size looks at the previous and next neighbours
        # when comparing textual similarity in predictions
        self.max_neighbours = max_neighbours

        self.ocr = MMOCRInferencer(det=det_model, det_weights=det_weights, rec=rec_model, rec_weights=rec_weights,
                                   device=device)




    """
    Using MMOCR inferencer, get the bounding box predictions and text recognition for each box
    """
    def parseOCR(self, image):
        out = self.ocr(image)
        preds = out['predictions'][0]
        polys = preds['det_polygons']
        pred_texts = preds['rec_texts']
        pred_probs = preds['rec_scores']

        """
        Using a dict form:
        
        index = dict(
            prediction:str,
            probability:float,
            box:[float, float, float, float],  #Initially prediction is a polygon but using a VOC box [xmin, ymin, xmax, ymax]
            matches:boolean
        ) 
        
        for all predictions
        """

        paired = {idx:{'prediction':item[0],'probability':item[2],'box':quad2voc(item[1]), 'matched':False} for idx,item in enumerate(list(zip(pred_texts, polys, pred_probs)))}

        paired = removeBelowAverageHeight(paired)

        return paired

    """
    Given the predictions and original text in a dict order, 
    Return the ordered (by word) prediction per word
    """
    def parsePreds(self, preds, txtOrder):
        #Order the pairs with accordance to (x-center, y-center) for each box

        orderedPair = orderPaired(preds)
        #Get the possible neighbours for all predicted boxes
        orderedPair = getNeighbours(orderedPair)

        #Get all possible sentence orders
        sentences = [txt['sent_order'] for txt in txtOrder.values()]
        sentences = list(set(sentences))

        if sentences[0] == -1:
            #If not sentence delimiters then do getLevMatched on all text
            finalPairs, txtOrderFinal = getLevMatched(orderedPair, txtOrder)
        else:
            #If not perform it on sentence by sentence
            finalPairs = {}
            txtOrderFinal = {}
            for sentence in sentences:
                try:
                    op_sent = {k:v for k,v in orderedPair if v['sent_order'] == sentence}
                    txt_sent = {k:v for k,v in txtOrder if v['sent_order'] == sentence}
                except:
                    continue

                op_sent, txt_sent = getLevMatched(op_sent, txt_sent)

                finalPairs.update(op_sent)
                txtOrderFinal.update(txt_sent)

        #Try generating the unknown boxes, if not possible for any single box then add ignore to UNK boxes
        try:
            final = generateUnknownBoxes(finalPairs, meta)
        except:
            final = addIgnore(finalPairs)

        return final

    """
    Perform generation on a image given text, this could be on a cropped image or an uncropped image
    """
    def fullParse(self, image, text):

        #If image is in string format, it is expected to be a path to image hence validity is checked
        if isinstance(image, str):
            try:
                openedImg = Image.open(image)
            except:
                raise FileNotFoundError(f"{image} does not exist")

        #If it is an object (almost all the time it is a PIL object), convert said image object to numpy array
        #since MMOCR only accepts paths, image names and or numpy arrays
        elif isinstance(image, object):
            image = np.array(image)

        #Get the text order
        txtOrder = getTextOrder(text, self.max_neighbours)

        try:
            preds = self.parseOCR(image)
            out = self.parsePreds(preds, txtOrder)
        except:
            out = {0:{'original':"UKN", "prediction":"UKN", "lev_dist":-1, "box":[],"sent_order":-1,"ignore":True}}

        return out

    """
    If there is a image with a localised textual section, the crop that section and perform generation on that crop
    then normalise the outputs back to original image
    """
    def croppedParse(self, image, text, bbox):
        """
        Assuming bounding box to be in VOC format (xmin, ymin, xmax, ymax)
        :param image:
        :param text:
        :param bbox:
        :return:
        """

        #Image is expected to be a path
        try:
            openedImg = Image.open(image)
        except:
            raise FileNotFoundError(f"{image} does not exist")

        #Get image size
        orig_width, orig_height = openedImg.size

        #unpack bounding box
        xmin, ymin, xmax, ymax = bbox

        #Crop the image with accordance to the bounding box
        crop_image = openedImg.crop(box=bbox)

        #Cropped image width and height
        crop_width, crop_height = crop_image.size

        #Parse through the croppped image
        out = self.fullParse(crop_image, text)

        #for all predictions, normalise non-empty bounding boxes to original image
        for key, value in out.items():
            if value["box"] != []:
                value["box"] = self.normalise2original(value["box"], (crop_width, crop_height), bbox)

        return out

    #TODO :: Need to check if it works
    def normalise2original(self, cropped_box, crop_size, crop_region_bbox):

        cr_width = crop_region_bbox[2] - crop_region_bbox[0]
        cr_height = crop_region_bbox[3] - crop_region_bbox[1]

        region_origin_x = crop_region_bbox[0]
        region_origin_y = crop_region_bbox[1]

        width_ratio = cr_width / crop_size[0]
        height_ratio = cr_height / crop_size[1]

        cropped_box[0] = (cropped_box[0] * width_ratio) + region_origin_x
        cropped_box[1] = (cropped_box[1] * height_ratio) + region_origin_y
        cropped_box[2] = (cropped_box[2] * width_ratio) + region_origin_x
        cropped_box[3] = (cropped_box[3] * height_ratio) + region_origin_y

        return cropped_box


    def __call__(self, image, texts, boxes=None):

        #Makes sure texts is a list or string
        assert isinstance(texts, (str, list)), "Expected texts to be a list of texts / a singular string for given image"

        #If boxes are provided, then make sure each box in the list of boxes is a list
        if not(boxes is None):
            assert any(isinstance(box, list) for box in boxes), f"Expected a list of bounding boxes, where the elements are of VOC format [xmin, ymin, xmax, ymax]\n" \
                                                                    f"but got {boxes} and {texts}"


        if isinstance(texts, str):
            texts = [texts]

        #Case 1: Full image and text for image is given with no need for cropping
        if (len(texts) == 1) and (boxes is None):
            return self.fullParse(image, texts[0])

        #Case 2: Cropping is needed, make sure for given text a bounding box exits and then
        #Perform cropped parse on each image, text and bounding box
        out = {}
        if len(texts) >= 1:
            assert not(boxes is None), "Given a list of texts, expecting a list of bounding boxes (sparse/dense)"
            assert (len(texts) == len(boxes)), "Expecting equal number of instances for boxes and texts"

            for text, box in zip(texts, boxes):
                out.update(self.croppedParse(image, text, box))

        return out

