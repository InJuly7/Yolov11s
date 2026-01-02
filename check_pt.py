from ultralytics import YOLO

model = YOLO("/root/autodl-tmp/model/yolov11/yolov11s.pt")

# 1) 打印整体对象信息
print(model)

# 2) 打印底层 PyTorch 网络结构（常用）
print(model.model)

# 3) Ultralytics 的结构/统计信息（有些版本支持 summary/info）
model.info(verbose=True)


# model.info()
# results = model("./ssd_horse.jpg")
# results[0].save("yolov11s-result.jpg")

# # Export to ONNX with static input size 1920x1080 and opset 12
# model.export(format='onnx',
#             simplify=True,
#             opset=12,
#             dynamic=False,
#             batch = 1,
#             imgsz=[640, 640])


# from ultralytics import YOLO
# import yaml

# pt = "/root/autodl-tmp/model/yolov11/yolov11s.pt"
# model = YOLO(pt)

# # 这个通常就是网络结构配置（dict）
# cfg = model.model.yaml
# print(cfg.keys())
# print(cfg)

# # 保存成 yaml 文件
# with open("yolov11s_extracted.yaml", "w", encoding="utf-8") as f:
#     yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

# print("saved -> yolov11s_extracted.yaml")


# from ultralytics import YOLO

# model = YOLO("/root/autodl-tmp/model/yolov11/yolov11s.pt")

# # 1) 更详细的info（不同版本参数名可能略有差异）
# model.info(verbose=True)

# # 2) 直接打印网络
# print(model.model)

# # 3) 逐层看（Ultralytics 的层通常有 f(来自哪些层), i(索引)等属性）
# net = model.model.model  # 一般是 ModuleList
# for i, m in enumerate(net):
#     f = getattr(m, "f", None)      # from
#     t = m.__class__.__name__       # type
#     np = sum(p.numel() for p in m.parameters())
#     print(f"{i:>3}  from={f!s:>8}  type={t:<25}  params={np}")
