import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import cv2
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from model.ViT.modeling import VisionTransformer, CONFIGS


class ViTFeatureExtraction(nn.Module):
    def __init__(self, img_resize=224):
        super(ViTFeatureExtraction, self).__init__()
        
        # self
        self.resize = (img_resize, img_resize)
        self.config = CONFIGS["ViT-B_16"]
        self.model = VisionTransformer(self.config, num_classes=1000, zero_head=False, img_size=self.resize[0], vis=True)
        self.model.load_from(np.load("./model/ViT/ViT-B_16-224.npz"))
        self.model.eval()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def preprocess(self, img_array):
        self.original_size = img_array.shape[:2]
        
        img_array = cv2.resize(img_array, self.resize)
        
        # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255
        img_tensor = (img_tensor - 0.5) / 0.5
        img_tensor = img_tensor.to(self.device)
        # print(f'img_tensor device: {img_tensor.device}')
        
        return img_array, img_tensor # torch.Size([3, 224, 224])
        
    def get_feature(self, img_array):
        # preprocess
        img_array, img_tensor = self.preprocess(img_array)
        with torch.no_grad():
            _, att_mat = self.model(img_tensor.unsqueeze(0))

        att_mat = torch.stack(att_mat).squeeze(1)

        # Average the attention weights across all heads.
        att_mat = torch.mean(att_mat, dim=1)

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1)).to(self.device)
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size()).to(self.device)
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
            
        # Attention from the output token to the input space.
        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
        # print(f'mask shape: {mask.shape}')
        mask = cv2.resize(mask / mask.max(), self.resize)[..., np.newaxis]
        result = (mask * img_array).astype("uint8") # numpy.ndarray
        
        return result # (224, 224, 3)
    
    def get_cls_token(self, img_array):
        # Preprocess
        _, img_tensor = self.preprocess(img_array)
        outputs = self.model(img_tensor)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        print(cls_embedding.shape)
        # with torch.no_grad():
        #     outputs = self.model(img_tensor.unsqueeze(0))
        
        # # Extract the CLS token (last hidden state)
        # last_hidden_state = outputs[0]
        # print(outputs)
        # cls_token_embedding = last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        
        # return cls_token_embedding.squeeze(0).detach().cpu().numpy()  # Shape: (hidden_size,)


img_path = './img/reacher_1.png'
img = Image.open(img_path).convert('RGB')
rgb_array = np.array(img)

vit = ViTFeatureExtraction()
attention_map = vit.get_feature(rgb_array)
print(vit.model.eval())
vit.get_cls_token(rgb_array)
# print(vit.get_cls_token(rgb_array))
# plt.imshow(vit.preprocess(rgb_array).permute(1,2,0).numpy())
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
        
        self.fc1 = nn.Linear(np.prod(self.input_dims), self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.CrossEntropyLoss()

    def imgarray_resize(self, img_array, width=50, height=50):
        resize_image = np.zeros(shape=(width,height,3), dtype=np.uint8)

        for W in range(width):
            for H in range(height):
                new_width = int( W * img_array.shape[0] / width )
                new_height = int( H * img_array.shape[1] / height )
                resize_image[W][H] = img_array[new_width][new_height]

        return resize_image
        
    def forward(self, img_array):
        img_array = self.imgarray_resize(img_array)
        img_array = torch.tensor(np.array([img_array]).astype(np.float32)).squeeze(0)
        # plt.imshow(img_array.numpy().astype(np.uint8))
        # plt.show()

        flatten = torch.flatten(img_array)
        feature = self.fc1(flatten)
        feature = F.relu(feature)
        feature = self.fc2(feature)
        feature = F.relu(feature) # torch.Size([256])

        return feature

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

# learning_rate = 0.001

# feature = MLPFeatureExtraction(learning_rate, (50, 50, 3))
# plt.imshow(feature.imgarray_resize(attention_map))
# plt.show()
# print(feature.forward(attention_map))