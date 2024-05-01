# from torchvision.models import resnet18
# from torchcam.methods import SmoothGradCAMpp, CAM
# from torchvision.io.image import read_image
# from torchvision.transforms.functional import normalize, resize, to_pil_image
# import matplotlib.pyplot as plt
# from torchcam.utils import overlay_mask
#
# img = read_image("src/ttt.png")
# input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).cuda()
# print(input_tensor)
# print(input_tensor.shape)
#
# model = resnet18(pretrained=True).to('cuda').eval()
# cam_extractor = SmoothGradCAMpp(model, 'layer4')
# # cam_extractor = CAM(model)
#
# out = model(input_tensor)
# print(out.shape)
#
# activation_map = cam_extractor(out[0].argmax().item(), out)
# print(activation_map[0].shape)
#
# # plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
# # plt.imshow(to_pil_image(activation_map[0].squeeze(0)).resize(to_pil_image(img).size)); plt.axis('off'); plt.tight_layout(); plt.show()
# # result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# # plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()

import re
import os

if __name__ == '__main__':
    dir = '1a/2b/3c/100.pt'
    num = int(re.findall(r'\d+', os.path.basename(dir))[0])
    print(num)
    print(type(num))