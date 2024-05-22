import cv2
import torch

import numpy as np

from PIL import Image
from torchvision import transforms

model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

img = Image.open('/home/kist/franka_panda_3_mppi/src/vit_explain/examples/both.png')
img.resize((224, 224))
input_tensor = transform(img).unsqueeze(0)

if torch.cuda.is_available():
    input_tensor = input_tensor.cuda()
    model = model.cuda()
    
print("Successed!")

def rollout(attentions, discatd_ratio, head_fusion):
    '''
    attentions: [1, 3, 197, 197] size의 tensor가 12개 담긴 list
    '''
    result = torch.eye(attentions[0].size(-1)).cuda()
    
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == 'mean':
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == 'max':
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == 'min':
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise 'Attention head fusion type Not supported'
            
            # Drop the lowest attentions, but don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discatd_ratio), dim = -1, largest = False)
            indices = indices[indices != 0]
            flat[0, indices] = 0
            
            I = torch.eye(attention_heads_fused.size(-1)).cuda()
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim = -1)
            
            result = torch.matmul(a, result) # [1, 197, 197]
            
    # look at the total attention between the class token,
    # and the image patches
    mask = result[0,0,1:] # [196, 196]
    
    # in case of 224 x 224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).cpu().numpy()
    mask = mask / np.max(mask)
    return mask # [14, 14]

class ViTAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion='mean', discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        
        for name, module in self.model.named_modules():
            print(name)
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                print(f"Hook registered for {name}")
                
        self.attentions = []
        
    def get_attention(self, module, input, output):
        print(f"Hook called for {module}")
        self.attentions.append(output.cpu())
        
    def __call__(self, input_tensor):
        self.attentions = []
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
        if not self.attentions:
            raise RuntimeError("No attentions were collected. Check the attention layer name and forward hook.")
        
        return rollout(self.attentions, self.discard_ratio, self.head_fusion)
    
def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)



model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
if torch.cuda.is_available():
    model = model.cuda()
grad_rollout = ViTAttentionRollout(model, discard_ratio=0.9, head_fusion='max')
mask = grad_rollout(input_tensor)

np_img = np.array(img)[:, :, ::-1]
mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
mask = show_mask_on_image(np_img, mask)

cv2.imshow("Attention Rollout", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
