from torchvision import models
from torchcam.methods import SmoothGradCAMpp
import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).cuda().eval()
cam_extractor = SmoothGradCAMpp(model, None)
print(cam_extractor.target_names)
frame = Image.open('ttt.png')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
obs = transform(frame).cuda().unsqueeze(0)
cam = model(obs)[0]
heatmap = cam_extractor(cam.argmax().item(), cam)[0]
print(heatmap.shape)
heatmap = transforms.Resize((obs.shape[-2], obs.shape[-1]))(heatmap)
print(heatmap.shape)
# mask = heatmap > 0.3
# mask_expanded = mask.repeat(obs.shape[0], 1, 1)
# mask_expanded = mask_expanded.to(obs.dtype)
# frame = (obs * mask_expanded).cpu().numpy()
heatmap = heatmap.repeat(obs.shape[0], 1, 1)
frame = (obs * 0.01 + heatmap * 0.99).cpu().numpy()
plt.imshow(frame[0].transpose(1, 2, 0))
plt.savefig('ttt_heatmap.png')