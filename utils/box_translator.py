from box_translator_utils import *

"""
A pipeline, that translates bounding boxes from one format to another
"""
class BoxTranslator:

    AVAIL_FORMATS = ["coco", "yolo", "quad", "voc"]

    def __init__(self, in_format=None, out_format=None):
        self.check_format(in_format)
        self.check_format(out_format)

        self.in_format = in_format
        self.out_format = out_format

    def check_format(self, format):
        assert not(format is None), "Expected a bounding box format"
        assert (format.lower() in self.AVAIL_FORMATS), f"Currently not supporting format: {format}, Available formats: {self.AVAIL_FORMATS}"

    def assign_in_format(self, format):
        self.check_format(format)

        self.in_format = format

    def assign_out_format(self, format):
        self.check_format(format)

        self.out_format = format

    def check(self, boxes):

        if any(isinstance(box, list) for box in boxes):
            return boxes
        elif any(isinstance(val, (int, float)) for val in boxes):
            return [boxes]
        else:
            raise ValueError(f"Expected List[List[Int, Float]] but got {type(boxes)}")

    def __call__(self, boxes):

        boxes = self.check(boxes)

        final_boxes = []

        for box in boxes:

            final_boxes.append(self.process(box))

        return final_boxes

    def process(self, box):

        if self.in_format.lower() == "voc":
            return self.processVOC(box)
        elif self.in_format.lower() == "yolo":
            return self.processYOLO(box)
        elif self.in_format.lower() == "coco":
            return self.processCOCO(box)
        elif self.in_format.lower() == "quad":
            return self.processQUAD(box)
        else:
            raise ValueError(f"Currently not supporting in_format: {self.in_format}")

    def processVOC(self, box):

        if self.out_format.lower() == "voc":
            return box
        elif self.out_format.lower() == "yolo":
            return voc2yolo(box)
        elif self.out_format.lower() == "coco":
            return voc2coco(box)
        elif self.out_format.lower() == "quad":
            return voc2quad(box)
        else:
            raise ValueError(f"Currently not supporting out_format: {self.out_format}")

    def processYOLO(self, box):

        if self.out_format.lower() == "voc":
            return yolo2voc(box)
        elif self.out_format.lower() == "yolo":
            return box
        elif self.out_format.lower() == "coco":
            return yolo2coco(box)
        elif self.out_format.lower() == "quad":
            return yolo2quad(box)
        else:
            raise ValueError(f"Currently not supporting out_format: {self.out_format}")

    def processQUAD(self, box):

        if self.out_format.lower() == "voc":
            return quad2voc(box)
        elif self.out_format.lower() == "yolo":
            return quad2yolo(box)
        elif self.out_format.lower() == "coco":
            return quad2coco(box)
        elif self.out_format.lower() == "quad":
            return box
        else:
            raise ValueError(f"Currently not supporting out_format: {self.out_format}")

    def processCOCO(self, box):

        if self.out_format.lower() == "voc":
            return coco2voc(box)
        elif self.out_format.lower() == "yolo":
            return coco2yolo(box)
        elif self.out_format.lower() == "coco":
            return box
        elif self.out_format.lower() == "quad":
            return coco2quad(box)
        else:
            raise ValueError(f"Currently not supporting out_format: {self.out_format}")