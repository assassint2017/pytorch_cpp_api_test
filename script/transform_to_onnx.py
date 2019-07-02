"""

将一个pytorch模型变成ONNX的通用格式脚本
"""

import torch
import torchvision

dummy_input = torch.randn(1, 3, 224, 224).cuda()

model = torchvision.models.resnet18(pretrained=False).cuda()
model.load_state_dict(torch.load('../model/resnet18-5c106cde.pth'))
model.eval()

torch.onnx.export(model, dummy_input, "../model/resnet18.onnx", verbose=True)