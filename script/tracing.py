import torch
import torchvision
from torchvision import transforms
from PIL import Image
from time import time
import numpy as np

# An instance of your model.
model = torchvision.models.resnet18(pretrained=False).cuda()
model.load_state_dict(torch.load('../model/resnet18-5c106cde.pth'))
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224).cuda()

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("../model/resnet18.pt")

# read image
image = Image.open('../data/dog.png').convert('RGB')
default_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
      ])
image = default_transform(image)
image = image.cuda()

# forward
output = traced_script_module(image.unsqueeze(0))
print(output.size())

# print top-5 predicted labels
labels = np.loadtxt('../data/synset_words.txt', dtype=str, delimiter='\n')

data_out = output[0].data.cpu().numpy()
sorted_idxs = np.argsort(-data_out)

for i,idx in enumerate(sorted_idxs[:5]):
  print('top-%d label: %s, score: %f' % (i, labels[idx], data_out[idx]))

# top-0 label: n02108422 bull mastiff, score: 17.966347
# top-1 label: n02093428 American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier, score: 13.665328
# top-2 label: n02109047 Great Dane, score: 13.340916
# top-3 label: n02093256 Staffordshire bullterrier, Staffordshire bull terrier, score: 12.602612
# top-4 label: n02108089 boxer, score: 11.998150