from ultralytics import YOLO
import cv2
import torch
from collections import OrderedDict

from model import YOLOV11S
from model import test_yolov11s
from utils import compare_tensor


class Config:
    model_path = "/root/autodl-tmp/model/yolov11/yolov11s.pt"


def analysis_official_weights(model_path):
    print("=" * 80)
    print("Official Model Parameters:")
    official_weights = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = official_weights["model"].state_dict()

    total_params = 0
    param_details = OrderedDict()

    for key, value in state_dict.items():
        num_params = value.numel()
        total_params += num_params
        param_details[key] = {"shape": value.shape, "dtype": value.dtype, "num_params": num_params}
        # print(f"key: {key}, shape: {value.shape}, dtype: {value.dtype}, params: {num_params}")

    print(f"\n总key数量: {len(state_dict)}")
    print(f"总参数量: {total_params:,}")
    print("=" * 80)

    return state_dict, param_details, total_params


def analysis_custom_weights():
    print("=" * 80)
    print("Model Parameters:")
    model = YOLOV11S().to(dtype=torch.float16, device="cpu")
    model.eval()

    state_dict = model.state_dict()
    total_params = 0
    param_details = OrderedDict()

    for key, value in state_dict.items():
        num_params = value.numel()
        total_params += num_params
        param_details[key] = {"shape": value.shape, "dtype": value.dtype, "num_params": num_params}
        # print(f"key: {key}, shape: {value.shape}, dtype: {value.dtype}, params: {num_params}")

    print(f"\n总key数量: {len(state_dict)}")
    print(f"总参数量: {total_params:,}")
    print("=" * 80)

    return state_dict, param_details, total_params


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


def compare_model_weights(model_path):
    print("=" * 80)
    print("Comparing Model Parameters:")
    print("=" * 80)

    custom_state, custom_details, custom_total = analysis_custom_weights()
    official_state, official_details, official_total = analysis_official_weights(model_path)

    # 同时遍历 custom_details, official_details 比较每个key的后缀是否相同
    for idx, (custom_key, official_key) in enumerate(zip(custom_details, official_details)):
        custom_suffix = extract_suffix(custom_key)
        official_suffix = extract_suffix(official_key)
        if custom_suffix != official_suffix:
            print(f"Key后缀不匹配: custom({custom_key} -> {custom_suffix}) vs official({official_key} -> {official_suffix})")
            return
        # 后缀相同，比较参数细节
        custom_param = custom_details[custom_key]
        official_param = official_details[official_key]
        if custom_param["num_params"] != official_param["num_params"]:
            print(f"line: {idx}")
            print(
                f"参数数量不匹配 for key '{custom_key}<==>{official_key}': custom({custom_param['num_params']}) vs official({official_param['num_params']})"
            )
            print(
                f"参数形状不匹配 for key '{custom_key}<==>{official_key}': custom{custom_param['shape']} vs official{official_param['shape']}"
            )
            return
        if custom_param["shape"] != official_param["shape"]:
            print(
                f"参数数量匹配 for key '{custom_key}<==>{official_key}': custom({custom_param['num_params']}) vs official({official_param['num_params']})"
            )
            print(
                f"参数形状不匹配 for key '{custom_key}<==>{official_key}': custom{custom_param['shape']} vs official{official_param['shape']}"
            )
            return

    # 如果字典key个数不相等，直接返回
    if len(custom_details) != len(official_details):
        print(f"Key数量不匹配: custom({len(custom_details)}) vs official({len(official_details)})")
        return

    print("=" * 80)
    print("All model parameters match successfully!")
    print("=" * 80)

def compare_weights_data(official_state_dict, custom_state_dict, rtol=1e-5, atol=1e-8):
    """
    比较两个模型权重字典的数据是否一致
    
    Args:
        official_state_dict: 官方模型的state_dict
        custom_state_dict: 自定义模型的state_dict
        rtol: 相对误差容限
        atol: 绝对误差容限
        verbose: 是否打印详细信息
    
    Returns:
        bool: 是否所有权重都匹配
    """    
    # 检查key数量是否一致
    if len(official_state_dict) != len(custom_state_dict):
        raise ValueError(f"Key数量不匹配: custom({len(custom_state_dict)}) vs official({len(official_state_dict)})")
    
    # 遍历比较每个权重
    for custom_key, official_key in zip(custom_state_dict.keys(), official_state_dict.keys()):
        # 验证后缀是否匹配
        custom_suffix = extract_suffix(custom_key)
        official_suffix = extract_suffix(official_key)
        
        if custom_suffix != official_suffix:
            raise ValueError(f"Key后缀不匹配: custom({custom_key} -> {custom_suffix}) vs " f"official({official_key} -> {official_suffix})")
        
        # 获取权重数据
        custom_weight = custom_state_dict[custom_key]
        official_weight = official_state_dict[official_key]
        
        # 转换为numpy数组（如果是torch.Tensor）
        if isinstance(custom_weight, torch.Tensor):
            custom_weight = custom_weight.detach().cpu()
        if isinstance(official_weight, torch.Tensor):
            official_weight = official_weight.detach().cpu()
        
        # 检查形状是否一致
        if custom_weight.shape != official_weight.shape:
            raise ValueError(f"形状不匹配 {custom_key}: custom{custom_weight.shape} vs official{official_weight.shape}")
        
        # 检查数据类型
        if custom_weight.dtype != official_weight.dtype:
           raise ValueError(f"数据类型不同 {custom_key}: custom({custom_weight.dtype}) vs official({official_weight.dtype})")
        
        # 比较数值是否接近
        is_close = compare_tensor(custom_weight, official_weight, dtype=custom_weight.dtype)
        
        if not is_close:
            raise ValueError(f"权重数据不匹配 {custom_key}")
    print("=" * 80)
    print("所有权重数据均匹配成功！")
    print("=" * 80)
    return True

def load_official_weights_to_custom_model(custom_model, official_state_dict, custom_details, official_details):
    """
    将官方权重加载到自定义模型中

    Args:
        custom_model: 自定义模型实例
        official_state_dict: 官方模型的state_dict
        custom_details: 自定义模型的参数详情
        official_details: 官方模型的参数详情

    Returns:
        加载权重后的模型
    """
    print("=" * 80)
    print("Loading Official Weights to Custom Model:")
    
    new_state_dict = OrderedDict()

    # 将官方权重映射到自定义模型的key
    custom_keys = list(custom_details.keys())
    official_keys = list(official_details.keys())

    if len(custom_keys) != len(official_keys):
        raise ValueError(f"Key数量不匹配: custom({len(custom_keys)}) vs official({len(official_keys)})")

    for custom_key, official_key in zip(custom_keys, official_keys):
        # 验证后缀是否匹配
        custom_suffix = extract_suffix(custom_key)
        official_suffix = extract_suffix(official_key)

        if custom_suffix != official_suffix:
            raise ValueError(f"Key后缀不匹配: custom({custom_key} -> {custom_suffix}) vs " f"official({official_key} -> {official_suffix})")

        # 将官方权重复制到新的state_dict中，使用自定义模型的key
        new_state_dict[custom_key] = official_state_dict[official_key].clone()

    # 加载权重到自定义模型
    custom_model.load_state_dict(new_state_dict)
    print(f"成功加载 {len(new_state_dict)} 个权重参数")
    print("=" * 80)

    return custom_model


def test_model_results():
    model_path = "/root/autodl-tmp/model/yolov11/yolov11s.pt"
    model = YOLO(model_path).to(dtype=torch.float16, device="cpu")
    model.eval()

    custom_model = YOLOV11S().to(dtype=torch.float16, device="cpu")
    custom_model.eval()

    official_state, official_details, _ = analysis_official_weights(model_path)
    custom_state, custom_details, _ = analysis_custom_weights()
    custom_model = load_official_weights_to_custom_model(custom_model, official_state, custom_details, official_details)
    custom_state = custom_model.state_dict()
    compare_weights_data(custom_state, official_state)

    torch.manual_seed(77)
    torch.cuda.manual_seed_all(77)
    x_golden = torch.randn(1, 3, 640, 640).to(dtype=torch.float16, device="cpu")
    
    torch.manual_seed(77)  # 重新设置种子
    torch.cuda.manual_seed_all(77)
    x = torch.randn(1, 3, 640, 640).to(dtype=torch.float16, device="cpu")
    compare_tensor(x, x_golden, dtype=torch.float16)
    
    with torch.no_grad():
        out_golden, _ = model.model(x)
    print(f"out_golden: {out_golden.shape}")


    with torch.no_grad():
        out, _ = custom_model(x)
    print(f"out: {out.shape}")

    compare_tensor(out, out_golden, dtype=torch.float16)


if __name__ == "__main__":
    config = Config()
    # compare_model_weights(config.model_path)
    test_model_results()
