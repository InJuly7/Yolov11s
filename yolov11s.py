import torch
import cv2
import numpy as np
from model import YOLOV11S


class YOLOv11s_Config:
    def __init__(self):
        self.img_path = "/root/program/YOLOv11-DeepSort/Yolov11/ssd_horse.jpg"
        self.resize_image_path = "/root/program/YOLOv11-DeepSort/Yolov11/resize_image.jpg"
        self.output_image_path = "/root/program/YOLOv11-DeepSort/Yolov11/yolov11s_result.jpg"
        self.model_path = "/root/autodl-tmp/model/yolov11/yolov11s_custom.pt"
        self.dst_h = 640
        self.dst_w = 640
        self.src_h = 0
        self.src_w = 0
        self.channel = 3
        self.padding = (144.0, 144.0, 144.0)
        self.class_nums = 80
        self.box_nums = 8400
        self.score_threshold = 0.45
        self.nms_threshold = 0.2


def xyxy2xywh(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def letterbox(image, config):
    """
    @func: 图像缩放与填充函数, 保持长宽比不变, 填充到目标尺寸
    @param: image: 输入图像
    @param: config: 配置参数
    @return: 处理后的图像, 缩放比例, 填充偏移
    """
    # 选择高度和宽度缩放比例中的较小值，确保图像完全适应目标尺寸
    # 满足 src_h * r <= dst_h 且 src_w * r <= dst_w
    r = min(config.dst_h / config.src_h, config.dst_w / config.src_w)
    # round 四舍五入函数
    # 计算缩放后但未填充的图像尺寸
    unpad_hw = (int(round(config.src_h * r)), int(round(config.src_w * r)))

    # 计算 左,上,右,下的 padding 宽度
    dh = (config.dst_h - unpad_hw[0]) / 2
    dw = (config.dst_w - unpad_hw[1]) / 2
    print(f"dh: {dh}, dw: {dw}")
    # 对于 dw xxx.0时候(dw*2为偶数) round(xxx.0 - 0.1) == round(xxx.0 + 0.1)
    # 对于 dw xxx.5时候(dw*2为奇数) round(xxx.5 - 0.1) + 1 == round(xxx.5 + 0.1)
    # 当需要的总padding为奇数时，无法平均分配到两侧，必须让一侧多1个像素。
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # 如果原始尺寸(宽,高)不等于计算出的新尺寸，则需要缩放
    if config.src_h != unpad_hw[0] or config.src_w != unpad_hw[1]:
        # 使用线性插值方法将图像缩放到unpad_hw尺寸
        # cv2.resize 需要 (width, height) 格式，所以要调换顺序
        image = cv2.resize(image, (unpad_hw[1], unpad_hw[0]), interpolation=cv2.INTER_LINEAR)
        print(f"Resized image to: {unpad_hw[0]}x{unpad_hw[1]}")
    # 在图像周围添加边框填充，使用常数填充方式，填充颜色为指定的color值
    # top, bottom, left, right 分别表示四个方向的填充宽度
    # cv2.BORDER_CONSTANT 表示使用常数值进行填充
    # value=config.padding 指定填充颜色
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=config.padding)  # add border
    print(f"Padded image to: {image.shape[0]}x{image.shape[1]}")
    # 返回处理后的图像和变换参数
    return image, r, (left, right, top, bottom)


def pre_process(image_path, config):
    """
    @func:  图像预处理函数, Resize + BGR2RGB + HWC2CHW + Normalized
    @param: image_path: 输入图像路径
    @return: 前处理后的图像张量, 缩放比例, 填充偏移
    """

    image = cv2.imread(image_path)  # BGR
    config.src_h = image.shape[0]
    config.src_w = image.shape[1]
    config.channel = image.shape[2]
    print(f"原图尺寸: {config.src_h}x{config.src_w}, 通道数: {config.channel}")
    # resize
    resize_image, scale_ratio, padding_offset = letterbox(image, config)
    print(f"Resize尺寸: {resize_image.shape[0]}x{resize_image.shape[1]}, 缩放比例: {scale_ratio}")
    cv2.imwrite(config.resize_image_path, resize_image)

    # torch.tensor 实现
    # 先将 numpy array 转换为 torch.tensor
    resize_image = torch.from_numpy(resize_image)
    # BGR2RGB, 使用 flip 在最后一个维度上翻转
    resize_image = torch.flip(resize_image, dims=[-1]).float()
    # HWC2CHW
    resize_image = resize_image.permute(2, 0, 1)
    # Normalized
    resize_image = resize_image / 255.0
    # 增加 batch 维度 NCHW
    resize_image = resize_image.unsqueeze(0)

    # numpy 实现
    # BGR2RGB ::-1 颠倒最后一个维度
    # resize_image = resize_image[:, :, ::-1].astype(dtype=np.float32)
    # HWC2CHW
    # resize_image = resize_image.transpose(2, 0, 1)
    # Normalized
    # resize_image = resize_image / 255.0
    return image, resize_image, scale_ratio, padding_offset


def probiou(box1, box2, eps=1e-7):
    """
    计算两个边界框IoU(Intersection over Union)
    """
    # 解包边界框坐标
    l1, t1, r1, b1 = box1.unbind(-1)
    l2, t2, r2, b2 = box2.unbind(-1)
    iou = 0.0
    # 计算交集区域的坐标
    inter_left = torch.max(l1, l2)
    inter_top = torch.max(t1, t2)
    inter_right = torch.min(r1, r2)
    inter_bottom = torch.min(b1, b2)

    # 计算交集的宽度和高度
    inter_width = torch.clamp(inter_right - inter_left, min=0)
    inter_height = torch.clamp(inter_bottom - inter_top, min=0)
    # 计算交集面积
    inter_area = inter_width * inter_height

    # 计算各个框的面积
    area1 = (r1 - l1) * (b1 - t1)
    area2 = (r2 - l2) * (b2 - t2)

    # 计算并集面积
    union_area = area1 + area2 - inter_area

    # 计算IoU
    iou = inter_area / (union_area + eps)
    return iou


def non_max_suppression(boxes, nms_thresh):
    """
    边界框的非极大值抑制(Non-Maximum Suppression)

    参数:
    boxes: 边界框列表, 每个框包含至少7个元素 [x, y, w, h, score, class_id, flag]
           其中前4个元素 [x, y, w, h] 是检测框的几何信息
           第5个元素是置信度分数,
           第6个元素是类别ID,
           第7个元素是标记位(初始为非-1的值)
    nms_thresh: NMS阈值, 当两个框的IoU大于此值时, 保留置信度更高的框

    返回:
    pred_boxes: 经过NMS筛选后保留的边界框列表
    """
    pred_boxes = []
    sort_boxes = boxes  # sorted(boxes, key=lambda x: x.score, reverse=True)
    for i in range(len(sort_boxes)):
        # NMS 剔除掉不满足条件的框
        if sort_boxes[i][6] != -1:
            for j in range(i + 1, len(sort_boxes), 1):
                ious = probiou(sort_boxes[i][:4], sort_boxes[j][:4])
                if ious > nms_thresh:
                    sort_boxes[j][6] = -1
            pred_boxes.append(sort_boxes[i])
    # for pred_box in pred_boxes:
    #     print(f"pred_box: {pred_box}")
    return pred_boxes


# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))


def post_process(output, config):
    output = output.squeeze(0)  # [1,num_boxes, boxes_width] -> [num_boxes, boxes_width]
    print(f"Output tensor shape: {output.shape}")
    print(f"Boxes nums: {output.shape[0]}, boxes_width: {output.shape[1]}, Class nums: {config.class_nums}")
    print("Bounding box threshold filtering")
    # 每行84个值的含义:
    # [cx, cy, w, h, class_0, class_1, ..., class_79]
    #  0   1   2  3  4        5             83
    output = output.transpose(-1, -2)  # 交换最后两个维度
    # 前4个: cx, cy, w, h [8400,4]
    # ... 表示选择所有前面的维度
    boxes = output[..., :4]
    # 后面80个: 类别分数 [8400,80]
    class_scores = output[..., 4 : 4 + config.class_nums]  # [8400,80]
    class_ids = torch.argmax(class_scores, dim=1)  # [8400]
    # 对每个样本 i，从 class_scores[i, :] 中取出 class_ids[i] 对应的那个类别分数，组成 scores[i]
    scores = torch.gather(class_scores, 1, class_ids.unsqueeze(1)).squeeze(1)  # [8400,1] -> [8400]
    mask = scores > config.score_threshold  # [8400] 布尔掩码
    # print(f"After filter boxes nums: {torch.sum(mask).item()}")
    # 布尔索引
    filtered_boxes = boxes[mask, :4]  # 过滤后的框 # [mask.sum(), 4]
    filtered_scores = scores[mask]
    filtered_class_ids = class_ids[mask]
    filtered_boxes = xywh2xyxy(filtered_boxes)  # 转换为 xyxy 形式
    flags = torch.ones_like(filtered_scores)
    candidate_box = torch.cat(
        [
            filtered_boxes,  # [N, 4]
            filtered_scores.unsqueeze(1),  # [N, 1]
            filtered_class_ids.unsqueeze(1).float(),  # [N, 1]
            flags.unsqueeze(1),  # [N, 1]
        ],
        dim=1,
    )
    print(f"Candidate box shape before NMS: {candidate_box.shape}")

    # 按分数排序
    if len(filtered_scores) > 0:
        sorted_indices = torch.argsort(filtered_scores, descending=True)
        candidate_box = candidate_box[sorted_indices]  # 高级索引
        # NMS 处理
        output = non_max_suppression(candidate_box, config.nms_threshold)
        output = torch.stack(output)  # list -> tensor
        print(f"After NMS boxes shape: {output.shape}")
    else:
        output = []
    return output


def draw_boxes(image, boxes, input_size, original_size, scale_ratio, padding_offset, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制多个检测框，并将坐标映射到原图大小。
    Args:
        image: 输入图像
        boxes: 一个包含多个框的列表，每个框是 (x, y, w, h, r)
        input_size: 输入图像的尺寸 (input_width, input_height)
        original_size: 原始图像的尺寸 (original_width, original_height)
        scale_ratio: letterbox缩放比例
        padding_offset: letterbox填充偏移 (pad_left, pad_top)
        color: 绘制框的颜色 (B, G, R)
        thickness: 框的线条宽度
    """
    original_height, original_width = original_size
    pad_left, pad_top = padding_offset
    print(f"原图尺寸: {original_height}x{original_width}, 缩放比例: {scale_ratio}, 填充: {padding_offset}")
    # 批量处理：先减去填充，再除以缩放比例
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale_ratio  # x坐标
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale_ratio  # y坐标
    for box in boxes:
        print(f"box: {box}")
    # 裁剪到原图范围
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_height)
    boxes_np = boxes.cpu().numpy().astype(int)

    for i, box in enumerate(boxes_np):
        x1, y1, x2, y2, scores, class_id, _ = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # 添加标签
        # if scores is not None and class_id is not None:
        #     label = f"{int(class_id)}: {scores:.2f}"
        #     cv2.putText(image, label, (x1, y1 - 5),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image


if __name__ == "__main__":
    config = YOLOv11s_Config()
    print("Pre-process")
    image, tensor, scale_ratio, padding_offset = pre_process(config.img_path, config)
    # CHW ==> NCHW, 扩展了一个维度
    print(f"Input Tensor Shape: {tensor.shape}")

    print("Inference")
    model = YOLOV11S().to(dtype=torch.float32, device="cpu")
    state_dict = torch.load(config.model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        output_tensor, _ = model(tensor)
    print(f"Output Tensor Shape: {output_tensor.shape}")

    print("Post-process")
    output = post_process(output_tensor, config)

    draw_boxes(
        image, output, (config.dst_h, config.dst_w), (config.src_h, config.src_w), scale_ratio, (padding_offset[0], padding_offset[2])
    )
    cv2.imwrite(config.output_image_path, image)
