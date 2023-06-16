

"""
Given a dictionary format of a COCO format, return VOC format bounding box
And optionally return the polygon, if stated
"""
def getPolygon(box):
    #Unpack data
    xmin,ymin,xmax,ymax = box

    #Calculate polygon
    x1,y1 = x_min, y_min
    x3,y3 = x_max, y_max

    x2,y2 = x_max, y_min
    x4,y4 = x_min, y_max

    poly = [x1,y1,x2,y2,x3,y3,x4,y4]

    return poly