import numpy as np


def Iou(box, boxes):
    box_area = box[2] * box[3]
    area = boxes[:, 2] * boxes[:, 3]

    w_min = np.where(box[0] < boxes[:, 0], box[2], boxes[:, 2])
    h_min = np.where(box[1] < boxes[:, 1], box[3], boxes[:, 3])
    w = w_min - np.abs(box[0] - boxes[:, 0])
    h = h_min - np.abs(box[1] - boxes[:, 1])

    w = np.where(w < 0, 0, w)
    h = np.where(h < 0, 0, h)

    inter = w * h
    ovr = np.true_divide(inter, (box_area + area - inter))

    return ovr






if __name__ == '__main__':

    boxes = np.array(
    [[ 69,359 ,50  ,36],
     [227, 382 , 56 , 43],
     [296, 305,  44 , 26],
     [353, 280 , 40 , 36],
     [885 ,377  ,63 , 41],
     [819 ,391  ,34 , 43],
     [727 ,342  ,37  ,31],
     [598 ,246 , 33 , 29],
     [740, 308  ,45,  33]])
    box =  [885 ,377 , 63 , 41]

    print(Iou(box, boxes))