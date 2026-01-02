def extract_suffix(key):
    """
    提取key的后缀部分

    规则:
    - 官方模型: model.数字.xxx -> xxx (提取第二个'.'之后的内容)
    - Backbone: Backbone.p数字.数字.xxx -> xxx (提取第三个'.'之后的内容)
    - head: head.h数字.xxx -> xxx (提取第二个'.'之后的内容)
    - Detect: Detect.xxx -> xxx (提取第一个'.'之后的内容)

    Args:
        key: 权重的key名称

    Returns:
        suffix: 提取的后缀部分
    """
    # 将key按'.'分割成部分
    # 如: "Backbone.p2.1.cv2.bn.weight" -> ['Backbone', 'p2', '1', 'cv2', 'bn', 'weight']
    parts = key.split(".")
    if key.startswith("Backbone."):
        # Backbone.p2.1.cv2.bn.weight -> cv2.bn.weight
        # 跳过前3个部分: ['Backbone', 'p2', '1']
        if len(parts) > 3:
            return ".".join(parts[3:])
        else:
            return key  # 异常情况,返回原key

    elif key.startswith("head."):
        # head.h1.m.0.cv2.bn.running_mean -> m.0.cv2.bn.running_mean
        # 跳过前2个部分: ['head', 'h1']
        if len(parts) > 2:
            return ".".join(parts[2:])
        else:
            return key

    elif key.startswith("Detect."):
        # Detect.cv3.2.0.0.bn.weight -> cv3.2.0.0.bn.weight
        # 跳过前1个部分: ['Detect']
        if len(parts) > 1:
            return ".".join(parts[1:])
        else:
            return key

    elif key.startswith("model."):
        # model.23.cv2.0.0.conv.weight -> cv2.0.0.conv.weight
        # 跳过前2个部分: ['model', '23']
        if len(parts) > 2:
            return ".".join(parts[2:])
        else:
            return key

    else:
        # 未知格式,返回原key
        return key


# 测试用例
if __name__ == "__main__":
    test_cases = [
        # 官方模型
        "model.23.cv2.0.0.conv.weight",
        "model.23.cv2.0.0.bn.weight",
        # Backbone
        "Backbone.p2.1.cv2.bn.running_var",
        "Backbone.p2.1.m.0.cv1.conv.weight",
        # head
        "head.h1.m.0.cv2.bn.bias",
        "head.h1.m.0.cv2.bn.running_mean",
        # Detect
        "Detect.cv3.2.0.0.conv.weight",
        "Detect.cv3.2.0.0.bn.weight",
    ]

    print("测试 extract_suffix 函数:\n")
    for key in test_cases:
        suffix = extract_suffix(key)
        print(f"原始: {key}")
        print(f"后缀: {suffix}\n")
