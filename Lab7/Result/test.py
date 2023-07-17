import torch
from modules import UNet_conditional
from main import Diffusion
from utils import get_data, setup_logging, plot_images, save_images, label_to_onehot
import json
import os 
from evaluator import evaluation_model
from PIL import Image
import torchvision
from tqdm import tqdm

default_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
label_dict = json.load(open('dataset/objects.json'))
test = json.load(open('dataset/test.json'))
new_test = json.load(open('dataset/new_test.json'))

exp_name = "DDPM_conditional_five_layers"

n=24
device = 'cuda:0'
model = UNet_conditional(num_classes=n).to(device)
ckpt = torch.load('./models/DDPM_conditional_five_layers/ckpt.pt')
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device)
# Turn the test label into one hot vector
# y = []
# for t in test:
#     label = torch.Tensor(label_to_onehot(t, label_dict)).to(device)
#     y.append(label)
# print(y)
# img = []
# for i in tqdm(range(len(y))):
#     x = diffusion.sample(model, 1, y[i], cfg_scale=3)
#     img_path = os.path.join("results", exp_name, "test", f"test_{i}.jpg")
#     save_images(x, img_path)
#     image = Image.open(img_path).convert('RGB')
#     image = default_transforms(image).to(device)
#     img.append(image.unsqueeze(0))
# y = torch.stack(y)
# # Start evaluation
# test = evaluation_model()
# img = torch.cat(img)
# print(test.eval(img, y))
# Store image
# x = diffusion.sample(model, 32, y, cfg_scale=3)
# img_path = os.path.join("results", exp_name, "test", f"test_{32}.jpg")
# save_images(x, img_path)

new_y = []
new_img = []
for t in new_test:
    label = torch.Tensor(label_to_onehot(t, label_dict)).to(device)
    new_y.append(label)
for i in tqdm(range(len(new_y))):
    x = diffusion.sample(model, 1, new_y[i], cfg_scale=3)
    img_path = os.path.join("results", exp_name, "new_test", f"new_test_{i}.jpg")
    save_images(x, img_path)
    image = Image.open(img_path).convert('RGB')
    image = default_transforms(image).to(device)
    new_img.append(image.unsqueeze(0))
new_y = torch.stack(new_y)
test = evaluation_model()
new_img = torch.cat(new_img)
print(test.eval(new_img, new_y))
# x = diffusion.sample(model, 32, new_y, cfg_scale=3)
# img_path = os.path.join("results", exp_name, "new_test", f"new_test_{32}.jpg")
# save_images(x, img_path)

