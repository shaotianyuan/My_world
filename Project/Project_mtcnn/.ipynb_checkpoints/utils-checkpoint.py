import numpy as np


def Iou(box, boxes):
    box_area = box[2] * box[3]
    area = boxes[:, 2] * boxes[:, 3]

    w_min = np.where(box[0] < boxes[:, 0], box[2], boxes[:, 2])
    h_min = np.where(box[1] < boxes[:, 1], box[3], boxes[:, 3])
    w = w_min - np.abs(box[0] - boxes[:, 0])
    h = h_min - np.abs(box[1] - boxes[:, 1])

    inter = w * h
    ovr = np.true_divide(inter, (box_area + area - inter))

    return ovr