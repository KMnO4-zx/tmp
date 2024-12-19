import math
import torch


def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from OBBs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(obb1, obb2, CIoU=False, HyperShapeIoU=False ,eps=1e-7):
    """
    Calculate probabilistic IoU between oriented bounding boxes.

    Implements the algorithm from https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 5), format xywhr.
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 5), format xywhr.
        CIoU (bool, optional): If True, calculate CIoU. Defaults to False.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-7.
        beta (float, optional): Scaling factor for the aspect ratio penalty. Defaults to 0.5.

    Returns:
        (torch.Tensor): OBB similarities, shape (N,).

    Note:
        OBB format: [center_x, center_y, width, height, rotation_angle].
        If CIoU is True, returns CIoU instead of IoU.
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    shapeIoU=False
    HyperShapeIoU=False
    # AngleEnhancedIoU = False
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    elif shapeIoU:
        # return iou - 0.5 * shape_cost
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
            
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)

        return iou - v * alpha - 0.01 * shape_cost  # CIoU
    elif HyperShapeIoU:
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        
        # 计算宽度和高度的相对差异
        omega_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omega_h = torch.abs(h1 - h2) / torch.max(h1, h2)

        # 计算对角线长度和面积
        d1 = torch.sqrt(w1.pow(2) + h1.pow(2))
        d2 = torch.sqrt(w2.pow(2) + h2.pow(2))
        A1 = w1 * h1
        A2 = w2 * h2

        # 计算对角线长度和面积的相对差异
        omega_d = torch.abs(d1 - d2) / torch.max(d1, d2)
        omega_A = torch.abs(A1 - A2) / torch.max(A1, A2)

        # 计算形状成本，包括宽度、高度、对角线长度和面积的差异
        hypershape_cost = (
            (1 - torch.exp(-omega_w)).pow(4) +
            (1 - torch.exp(-omega_h)).pow(4) +
            (1 - torch.exp(-omega_d)).pow(4) +
            (1 - torch.exp(-omega_A)).pow(4)
        )
        return iou - v * alpha - 0.1 * hypershape_cost
    return iou  # 返回形状为 (N,)


if __name__ == "__main__":
    # 定义两个 OBBs
    obb1 = torch.tensor([[0.0, 0.0, 2.0, 4.0, 0.0]])  # [x, y, w, h, r=0°]
    obb2 = torch.tensor([[5.0, 5.0, 6.0, 2.0, math.radians(45)], [0.0, 0.0, 6.0, 2.0, 0.0], [0.0, 0.0, 2.0, 4.0, 0.0]])  # [x, y, w, h, r=45°]
    print("OBB1:", obb1)
    print("OBB2:", obb2)

    # 调用 _get_covariance_matrix
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    # 计算概率IoU
    iou = probiou(obb1, obb2, CIoU=False)
    print("ProbIoU:", iou)

    # 计算CIoU
    ciou = probiou(obb1, obb2, CIoU=True)
    print("CIoU:", ciou)

    # 计算HyperShapeIoU
    HyperShapeIoU = probiou(obb1, obb2, HyperShapeIoU=True)
    print("HyperShapeIoU:", ciou)
