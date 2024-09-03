import torch
path = r"D:\上海cv\SuperGlobal\finetune\model\model_state.pth"
x = torch.load(path)
print(x)