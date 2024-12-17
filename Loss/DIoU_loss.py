import torch


def diou_loss(box1, box2, x1y1x2y2=True):
    '''
    :param box1: 一个 gt bbox， 尺寸为 (4)
    :param box2: 多个 predicted bbox， 尺寸为 (n, 4)
    :param x1y1x2y2: 坐标形式是否为 (xmin, ymin, xmax, ymax)
    :return: 返回 box2 与 bbox1 的 IoU
    '''

    box2 = box2.t()

    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # 将坐标形式由 (cx, cy, w, h) 转换为 (xmin, ymin, xmax, ymax)
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + 1e-16

    # iou
    iou = inter / union

    # external rectangle box
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

    # external rectangle box area squared
    c2 = cw ** 2 + ch ** 2 + 1e-16

    # centerpoint distance squared
    rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4

    return 1 - iou + rho2 / c2


if __name__ == '__main__':
    box1 = torch.tensor([1.0, 2.0, 5.0, 6.0])  # 一个gt bbox的坐标
    box2 = torch.tensor([[1.0, 2.0, 4.0, 5.0], [3.0, 4.0, 7.0, 8.0]])  # 2个预测的bbox的坐标
    
    loss = diou_loss(box1, box2)
    print(loss)