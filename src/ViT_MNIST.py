import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn, optim
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import random

# batch_size = 8
# channel = 3
# h, w = (224, 224)
# x = torch.randn(8, 3, 224, 224)
# plt.imshow(x[0].permute(1,2,0).numpy()) # (C, H, W) to (H, W, C)
# plt.show()
# print('x:', x.shape) # x: torch.Size([8, 3, 224, 224])

## for patch embedding
'''
    [batch] x [channel] x [height] x [width]의 구조를 가진 image를 
    다음과 같은 vector로 embedding해주어야 함

    -> [batch] x [n] x ([p] x [p] x [channel])
        - [p]: patch size
        - [n]: number of patch size = [height] x [width] / ([p] x [p])
'''

'''
## einops를 사용해서 image를 patch로 나누고 flatten하는 과정
patch_size = 16 # 16 pixels
patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1 = patch_size, s2 = patch_size)
print('patches:', patches.shape) # patches: torch.Size([8, 196, 768])


## ViT: kernel size와 stride size를 patch size로 갖는 Conv2D를 이용해서 flatten
patch_size = 16
in_channels = 3
emb_size = 768
img_size = 224

projection = nn.Sequential(
    # using a conv layer instead of a linear one -> performance gains
    nn.Conv2d(in_channels, emb_size, kernel_size = patch_size, stride = patch_size),
    Rearrange('b e (h) (w) -> b (h w) e'),
)

# image를 patch로 나누고, flatten
projected_x = projection(x)
print('projected_x:', projected_x.shape) # projected_x: torch.Size([8, 196, 768])

# cls_token과 pos encoding parameter 정의
cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
positions = nn.Parameter(torch.randn((img_size // patch_size)**2 + 1, emb_size))
print('cls shape:', cls_token.shape) # cls shape: torch.Size([1, 1, 768])
print('pos shape:', positions.shape) # pos shape: torch.Size([197, 768])

# cls_token을 반복하여 batch_size의 크기와 맞춰줌
batch_size = 8
cls_tokens = repeat(cls_token, '() n e -> b n e', b = batch_size)
print('repeated cls shape:', cls_tokens.shape) # repeated cls shape: torch.Size([8, 1, 768])

# cls_token과 projected_x를 concatenate (배열 합치기)
cat_x = torch.cat([cls_tokens, projected_x], dim = 1)

# position encoding을 더해줌
cat_x += positions
print('output:', cat_x.shape) # output: torch.Size([8, 197, 768])
'''

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 7, emb_size: int = 768, img_size: int = 28):
    # def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))
        self.feature = None # to store features
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        self.feature = x # save the features
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b = b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions

        return x
    
# x = PatchEmbedding()(x)

## for multi head attention
'''
# linear projection (q, k, v)
emb_size = 768
num_heads = 8

keys = nn.Linear(emb_size, emb_size)
queries = nn.Linear(emb_size, emb_size)
values = nn.Linear(emb_size, emb_size)
print('keys:', keys) # keys: Linear(in_features=768, out_features=768, bias=True)
print('queries:', queries) # queries: Linear(in_features=768, out_features=768, bias=True)
print('values:', values) # values: Linear(in_features=768, out_features=768, bias=True)

# multi-head
queries = rearrange(queries(x), "b n (h d) -> b h n d", h = num_heads)
keys = rearrange(keys(x), "b n (h d) -> b h n d", h = num_heads)
values = rearrange(values(x), "b n (h d) -> b h n d", h = num_heads)
print('queries shape:', queries.shape) # queries shape: torch.Size([8, 8, 197, 96])
print('keys shape:', keys.shape) # keys shape: torch.Size([8, 8, 197, 96])
print('values shape:', values.shape) # values shape: torch.Size([8, 8, 197, 96])

# queries x keys
energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
print('energy:', energy.shape) # energy: torch.Size([8, 8, 197, 197])

# get attention score
scaling = emb_size ** (1/2)
att = F.softmax(energy, dim=-1) / scaling
print('att:', att.shape) # att: torch.Size([8, 8, 197, 197])

# attention score x values
out = torch.einsum('bhal, bhlv -> bhav', att, values)
print('out:', out.shape) # out: torch.Size([8, 8, 197, 96])

# rearrange to emb_size
out = rearrange(out, 'b h n d -> b n (h d)')
print('out2:', out.shape) # out2: torch.Size([8, 197, 768])
'''

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 6, **kwargs):
    # def __init__(self, depth: int = 12, **kwargs):
        # stack encoder block amount of depth
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 10):
    # def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))
        
class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 1, # 3,
                patch_size: int = 7, # 16,
                emb_size: int = 768,
                img_size: int = 28, # 224,
                depth: int = 6, # 12,
                n_classes: int = 10, # 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

# patches_embedded = PatchEmbedding()(x)
# print('Patches embedded shape:', patches_embedded.shape)
# print('MultiHeadAttention output shape:', MultiHeadAttention()(patches_embedded).shape)
# print('TransformerEncoderBlock output shape:', TransformerEncoderBlock()(patches_embedded).shape)

# model = ViT()
# print(model)
# summary(model, (3, 224, 224), device='cpu')

# Prepare the MNIST dataset
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate the model, loss function and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print('device:', device)

# Initialize lists to store loss and accuracy
train_losses = []
test_losses = []
test_accuracies = []

# Training function
def train(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100}')
            running_loss = 0.0
    train_losses.append(running_loss / len(loader))

# Testing function
def test(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_losses.append(test_loss / len(loader))
    test_accuracies.append(accuracy)
    print(f'Test Loss: {test_loss / len(loader)}, Accuracy: {accuracy}%')
    print("outputs:", outputs)
    print(outputs.shape)
    print(outputs.size)
    plt.imshow(outputs.cpu())
    plt.show()

# Function to visualize predictions
def visualize_predictions(model, dataset):
    model.eval()
    images, labels = next(iter(dataset))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()

    # choice randomly: 5
    indices = random.sample(range(len(images)), 5)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for idx in range(5):
        axes[idx].imshow(images[indices[idx]].squeeze(), cmap='gray')
        axes[idx].set_title(f'True: {labels[indices[idx]]}\nPred: {predicted[indices[idx]]}')
        axes[idx].axis('off')
    plt.show()

num_epochs = 20

# Training and testing the model
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)
    test(model, test_loader, criterion)
    visualize_predictions(model, test_loader)  # Visualize predictions after each epoch
    
# Plot the results
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 2, 2)
plt.plot(epochs, test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy over Epochs')

plt.show()