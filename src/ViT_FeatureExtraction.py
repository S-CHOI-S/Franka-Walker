import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from model.ViT.modeling import VisionTransformer, CONFIGS


class ViTFeatureExtraction():
    def __init__(self):
        super(ViTFeatureExtraction, self).__init__()
        
        # self
        self.config = CONFIGS["ViT-B_16"]
        self.model = VisionTransformer(self.config, num_classes=1000, zero_head=False, img_size=224, vis=True)
        self.model.load_from(np.load("./model/ViT/ViT-B_16-224.npz"))
        self.model.eval()
        
    def preprocess(self, img_array):
        self.original_size = img_array.shape[:2]
        
        img_array = cv2.resize(img_array, (224, 224))

        # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        img_tensor = (img_tensor - 0.5) / 0.5
        
        return img_tensor # torch.Size([3, 224, 224])
        
    def get_feature(self, img_array):
        # preprocess
        img_tensor = self.preprocess(img_array)
        _, att_mat = self.model(img_tensor.unsqueeze(0))

        att_mat = torch.stack(att_mat).squeeze(1)

        # Average the attention weights across all heads.
        att_mat = torch.mean(att_mat, dim=1)

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
            
        # Attention from the output token to the input space.
        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), self.original_size)[..., np.newaxis]
        result = (mask * img_array * 255).astype("uint8") # numpy.ndarray
        
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

        # ax1.set_title('Original')
        # ax2.set_title('Attention Map')
        # _ = ax1.imshow(img_array)
        # _ = ax2.imshow(result)
        # plt.show()
        
        return result

# def rgb2gray(rgb_array):
#     return np.dot(rgb_array[...,:3], [0.2989, 0.5870, 0.1140])

img_path = './img/reacher_1.png'
img = Image.open(img_path).convert('RGB')
rgb_array = np.array(img)
print(rgb_array.shape)

# gray_array = rgb2gray(rgb_array)
# gray_image_expanded = np.expand_dims(gray_array, axis=-1)

# fig, axes = plt.subplots(1, 2)

# axes[0].imshow(rgb_array)
# axes[1].imshow(gray_image_expanded)
# plt.show()

# print(gray_image_expanded.shape)

# vit = ViTFeatureExtraction()
# plt.imshow(vit.get_feature(gray_image_expanded))
# plt.show()

class MLPFeatureExtraction(nn.Module):
    def __init__(self, learning_rate, input_dims, fc1_dims=256, fc2_dims=256,
                 name='feature', chkpt_dir='tmp/sac'):
        super(MLPFeatureExtraction, self).__init__()
        
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'sac')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        
    def forward(self, img_array):
        img_array = torch.Tensor(np.array([img_array]).astype(np.float32)).squeeze(0).to(self.device)
        
        # flatten 먼저!
        # prob = self.fc1(img_array)
        
        return print("HERE!", img_array.shape)






feature = MLPFeatureExtraction(0.1, (10, 10, 3))
feature.forward(rgb_array)