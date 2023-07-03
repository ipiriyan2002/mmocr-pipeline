from mmocr.apis import MMOCRInferencer
from bbox_gen.utils.box_processing import *
from bbox_gen.utils.ocr_processing import *
from bbox_gen.utils.text_processing import *
from PIL import Image
from utils.box_translator_utils import *
import os
import numpy as np

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
        
        self.max_neighbours = max_neighbours

        self.ocr = MMOCRInferencer(det=det_model, det_weights=det_weights, rec=rec_model, rec_weights=rec_weights,
                                   device=device)


    def parseOCR(self, image):
        out = self.ocr(image)
        preds = out['predictions'][0]
        polys = preds['det_polygons']
        pred_texts = preds['rec_texts']
        pred_probs = preds['rec_scores']

        paired = {idx:{'prediction':item[0],'probability':item[2],'box':quad2voc(item[1]), 'matched':False} for idx,item in enumerate(list(zip(pred_texts, polys, pred_probs)))}

        paired = removeBelowAverageHeight(paired)

        return paired

    def parsePreds(self, preds, txtOrder):
        orderedPair = orderPaired(preds)
        orderedPair = getNeighbours(orderedPair)

        sentences = [txt['sent_order'] for txt in txtOrder.values()]
        sentences = list(set(sentences))

        if sentences[0] == -1:
            finalPairs, txtOrderFinal = getLevMatched(orderedPair, txtOrder)
        else:
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

        try:
            final = generateUnknownBoxes(finalPairs, meta)
        except:
            final = addIgnore(finalPairs)

        return final

    def fullParse(self, image, text):

        if isinstance(image, str):
            try:
                openedImg = Image.open(image)
            except:
                raise FileNotFoundError(f"{image} does not exist")
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

    def croppedParse(self, image, text, bbox):
        """
        Assuming bounding box to be in VOC format (xmin, ymin, xmax, ymax)
        :param image:
        :param text:
        :param bbox:
        :return:
        """

        try:
            openedImg = Image.open(image)
        except:
            raise FileNotFoundError(f"{image} does not exist")

        orig_width, orig_height = openedImg.size

        xmin, ymin, xmax, ymax = bbox

        crop_image = openedImg.crop(box=bbox)

        crop_width, crop_height = crop_image.size

        out = self.fullParse(crop_image, text)

        for key, value in out.items():
            if value["box"] != []:
                value["box"] = self.normalise2original(value["box"], (orig_width, orig_height), (crop_width, crop_height), bbox)

        return out

    def normalise2original(self, cropped_box, og_size, crop_size, crop_bbox):
        og_xy = (0, 0)
        crop_xy = (crop_bbox[0], crop_bbox[1])

        offset_x = crop_xy[0] - og_xy[0]
        offset_y = crop_xy[1] - og_xy[1]

        width_ratio = og_size[0] / crop_size[0]
        height_ratio = og_size[1] / crop_size[1]

        cropped_box = voc2coco(cropped_box)

        cropped_box[0] = (cropped_box[0] * width_ratio) + offset_x
        cropped_box[1] = (cropped_box[1] * height_ratio) + offset_y
        cropped_box[2] = (cropped_box[2] * width_ratio)
        cropped_box[3] = (cropped_box[3] * height_ratio)

        cropped_box = coco2voc(cropped_box)

        return cropped_box

    def __call__(self, image, texts, boxes=None):


        assert isinstance(texts, (str, list)), "Expected texts to be a list of texts / a singular string for given image"

        if not(boxes is None):
            assert any(isinstance(box, list) for box in boxes), f"Expected a list of bounding boxes, where the elements are of VOC format [xmin, ymin, xmax, ymax]\n" \
                                                                    f"but got {boxes} and {texts}"


        if isinstance(texts, str):
            texts = [texts]

        if (len(texts) == 1) and (boxes is None):
            return self.fullParse(image, texts[0])

        out = {}
        if len(texts) >= 1:
            assert not(boxes is None), "Given a list of texts, expecting a list of bounding boxes (sparse/dense)"
            assert (len(texts) == len(boxes)), "Expecting equal number of instances for boxes and texts"

            for text, box in zip(texts, boxes):
                out.update(self.croppedParse(image, text, box))

        return out

