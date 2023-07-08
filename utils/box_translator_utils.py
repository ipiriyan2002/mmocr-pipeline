#Box Translators

#To PASCAL VOC format [xmin, ymin, xmax, ymax]


def quad2voc(box):

    x1,y1,x2,y2,x3,y3,x4,y4 = box

    x_min = min([x1,x2,x3,x4])
    y_min = min([y1,y2,y3,y4])
    x_max = max([x1,x2,x3,x4])
    y_max = max([y1,y2,y3,y4])

    return [x_min,y_min,x_max,y_max]

def coco2voc(box):

    xmin, ymin, w, h = box

    return [xmin, ymin, xmin+w, ymin+h]

def yolo2voc(box):

    xc,yc,w,h = box

    width_ratio = (w-1) / 2
    height_ratio = (h-1) / 2

    return [xc - width_ratio, yc - height_ratio, xc + width_ratio, yc + height_ratio]

#To quad format [x1,y1,x2,y2,x3,y3,x4,y4]

def voc2quad(box):

    xmin,ymin,xmax,ymax = box

    return [xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]

def coco2quad(box):

    xmin, ymin, w, h = box

    xmax = xmin + w
    ymax = ymin + h

    return [xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]

def yolo2quad(box):

    xc,yc,w,h = box

    width_ratio = (w-1) / 2
    height_ratio = (h-1) / 2

    xmin,ymin,xmax,ymax = [xc - width_ratio, yc - height_ratio, xc + width_ratio, yc + height_ratio]

    return [xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]

# To COCO format [xmin, ymin, w, h]

def voc2coco(box):

    xmin,ymin,xmax,ymax = box

    w = (xmax - xmin) + 1
    h = (ymax - ymin) + 1

    return [xmin, ymin, w, h]

def quad2coco(box):

    x1,y1,x3,y3 = quad2voc(box)

    w = (x3 - x1) + 1
    h = (y3 - y1) + 1

    return [x1,y1,w,h]

def yolo2coco(box):

    xc,yc,w,h = box

    width_ratio = (w-1) / 2
    height_ratio = (h-1) / 2

    return [xc - width_ratio, yc - height_ratio, w, h]

#To YOLO format [x center, y center, w, h]

def voc2yolo(box):

    xmin,ymin,xmax,ymax = box

    w = (xmax - xmin) + 1
    h = (ymax - ymin) + 1

    width_ratio = (w-1) / 2
    height_ratio = (h-1) / 2

    return [xmin + width_ratio, ymin + height_ratio, w, h]

def quad2yolo(box):

    x1,y1,x3,y3 = quad2voc(box)

    w = (x3 - x1) + 1
    h = (y3 - y1) + 1

    width_ratio = (w-1) / 2
    height_ratio = (h-1) / 2

    return [x1 + width_ratio, y1 + height_ratio, w, h]

def coco2yolo(box):

    xmin, ymin, w, h = box

    width_ratio = (w-1) / 2
    height_ratio = (h-1) / 2

    return [xmin + width_ratio, ymin + height_ratio, w, h]

"""
#To polygon

def voc2Polygon(box):
    #Unpack data
    xmin,ymin,xmax,ymax = box

    #Calculate polygon
    x1,y1 = x_min, y_min
    x3,y3 = x_max, y_max

    x2,y2 = x_max, y_min
    x4,y4 = x_min, y_max

    poly = [x1,y1,x2,y2,x3,y3,x4,y4]

    return poly

def coco2Polygon(box):
    #To VOC format from COCO format
    box = coco2voc(box)

    return voc2Polygon(box)

def yolo2Polygon(box):
    #To VOC format from YOLO format
    box = yolo2voc(box)

    return voc2Polygon(box)

def quad2Polygon(box):
    #To VOC format from QUAD format
    box = quad2voc(box)

    return voc2Polygon(box)
    
"""