import torch
import sys
import numpy as np
from PIL import Image
import os


class AttentionPatchExtractor:
    def __init__(self, model, processor, num_patches=5):
        self.model = model
        self.processor = processor
        self.attn_weights = None
        self.num_patches = num_patches
        
    def extract_topk(self, input_pixels):
        self.model.eval()
        with torch.no_grad():
            vision_outputs = self.model.vision_model(input_pixels, output_attentions = True, return_dict=True)
            
        self.attn_weights = vision_outputs.attentions[-1]  # Get the last attention layer's weights
        avg_attention = self.attn_weights.mean(dim=1) #batch_size, seq_len, seq_len
        cls_attention = avg_attention[:, 0, 1:]
        _, indices = torch.topk(cls_attention, self.num_patches, dim=1)
        return indices
    
    def get_patch_coords(self, indices):
        image_size = 224
        patch_size = 32
        num_patches_per_side = image_size // patch_size
        
        
        batch_size = len(self.attn_weights)
        patch_coords = np.zeros((batch_size, self.num_patches, 4), dtype=np.uint8)
        for batch_idx in range(batch_size):
            for i, patch_idx in enumerate(indices[batch_idx]):
                patch_number = patch_idx.item()
                
                row = patch_number // num_patches_per_side
                col = patch_number % num_patches_per_side
                
                # Get pixel coordinates
                left, top = col * patch_size, row * patch_size
                right, bottom = left + patch_size, top + patch_size
                patch_coords[batch_idx, i, :] = np.array([left, top, right, bottom])
        
        return patch_coords
    
    def get_patch_mask(self, input_pixels):
        selected_patch_indices = self.extract_topk(input_pixels)
        patch_coords = self.get_patch_coords(selected_patch_indices)
        batch_size = input_pixels.shape[0]
        mask = torch.zeros((batch_size, 1, 224, 224), 
                        dtype=torch.float32, 
                        device=input_pixels.device)
        
        # Fill mask using coordinates
        for b in range(batch_size):
            for (left, top, right, bottom) in patch_coords[b]:
                # Convert to tensor slice indices
                left = int(left)
                top = int(top)
                mask[b, 0, top:bottom, left:right] = 1.0
        
        return mask

        
        

##testing

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, parent_dir)
# from models.discriminator import CLIPSVMDiscriminator

# import matplotlib.pyplot as plt

# def plot_attention_patches(image, coords_list):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     ax = plt.gca()
    
#     for (left, top, right, bottom) in coords_list:
#         rect = plt.Rectangle((left, top), right-left, bottom-top, 
#                            linewidth=2, edgecolor='red', facecolor='none')
#         ax.add_patch(rect)
    
#     plt.show()

# discriminator = CLIPSVMDiscriminator()
# model = discriminator.model
# processor = discriminator.processor

# analyzer = AttentionPatchExtractor(model, processor)

# image_dir = "../images/"
# batch_paths = [os.path.join(image_dir, i) for i in os.listdir(image_dir)]
# batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
# inputs = processor(images=batch_images, return_tensors="pt").to(model.device)

# mask = analyzer.get_patch_mask(inputs.pixel_values)


# def apply_and_visualize_mask(image, mask_tensor):
#     """Apply mask overlay to image and display"""
#     # Convert mask to numpy array
#     mask_np = mask_tensor.squeeze().cpu().numpy()
    
#     # Convert image to 224x224 RGB
#     img_processed = processor.feature_extractor(image, return_tensors="pt").pixel_values.squeeze(0)
#     img_processed = img_processed.permute(1, 2, 0).numpy().astype(np.uint8)
    
#     # Create red overlay
#     overlay = np.zeros_like(img_processed)
#     overlay[..., 0] = 255  # Red channel
    
#     # Apply mask with alpha blending
#     alpha = 0.3
#     masked_img = np.where(mask_np[..., None], 
#                          (1 - alpha) * img_processed + alpha * overlay,
#                          img_processed)
#     if masked_img.dtype == float:
#         masked_img = np.clip(masked_img, 0, 1)
#     else:
#         masked_img = np.clip(masked_img, 0, 255).astype(np.uint8)
    
#     plt.imshow(masked_img)
#     plt.axis('off')
#     plt.show()

# print(batch_paths[0])
# # Test with first image
# first_image = batch_images[0]
# first_mask = mask[0].unsqueeze(0)  # Get mask for first image
# apply_and_visualize_mask(first_image, first_mask)
        