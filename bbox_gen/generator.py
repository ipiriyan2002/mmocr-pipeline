from mmocr.apis import MMOCRInferencer
from bbox_gen.utils.box_processing import *
from bbox_gen.utils.ocr_processing import *
from bbox_gen.utils.text_processing import *
from PIL import Image

class MMOCRBoxGenerator:

    def __init__(self, det_model="dbnet", det_weights=None, rec_model="satrn", rec_weights=None, device="cuda:0", save_dir="./box_gen/", max_neighbours=5):

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

        self.ocr = MMOCRInferencer(det=det_model, det_weights=det_weights, rec=rec_model, rec_weights=rec_weights, device=device)


    def parseOCR(self, image):
        out = self.ocr(img)
        preds = out['predictions'][0]
        polys = preds['det_polygons']
        pred_texts = preds['rec_texts']
        pred_probs = preds['rec_scores']

        paired = {idx:{'prediction':item[0],'probability':item[2],'box':vocBOX(item[1]), 'matched':False} for idx,item in enumerate(list(zip(pred_texts, polys, pred_probs)))}

        paired = removeBelowAverageHeight(paired)

        return paired

    def parsePreds(self, preds, txtOrder):
        orderedPair = orderPaired(preds)
        orderedPair = getNeighbours(orderedPair)
        orderedPair, txtOrder = getLevMatched(orderedPair, txtOrder)
        try:
            final = generateUnknownBoxes(orderedPair, meta)
        except:
            final = addIgnore(orderedPair)

        return final

    def fullParse(self, image, text):

        try:
            openedImg = Image.open(image)
        except:
            raise FileNotFoundError(f"{image} does not exist")

        #Get the text order
        txtOrder = getTextOrder(text, self.max_neighbours)

        try:
            preds = self.parseOCR(image)
            out = self.parsePreds(preds, txtOrder)
        except:
            out = {0:{'original':"UKN", "prediction":"UKN", "lev_dist":-1, "box":[],"sent_order":-1,"ignore":True}}

        return out

        


    def __call__(self, image, texts, boxes=None):

        assert isinstance(texts, (str, list)), "Expected texts to be a list of texts / a singular string for given image"

        if isinstance(texts, list) and len(texts) > 1:
            assert not(boxes is None), "Given a list of texts, expecting a list of bounding boxes (sparse/dense)"
        else:
            pass


        return self.fullParse(image, texts)

