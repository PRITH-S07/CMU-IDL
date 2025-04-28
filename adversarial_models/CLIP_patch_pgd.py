import torch
import torch.nn as nn
from torchattacks.attack import Attack

class CLIPPatchPGDAttack(Attack):
    def __init__(self, model, svm, eps=8 / 255, alpha=2/255, steps=10, 
                 patch_size=16, num_patches=5, random_start=True):
        super().__init__("PatchPGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.random_start = random_start
        self.supported_mode = ['default']
        self.svm_weights = torch.FloatTensor(svm.coef_[0])
        self.svm_bias = torch.tensor(svm.intercept_[0])

    
    def get_logits(self, inputs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)

        # Get image features from the vision model
        vision_outputs = self.model.vision_model(inputs)
        image_features = vision_outputs.last_hidden_state[:, 0, :]

        return image_features
        
    def create_patch_mask(self, gradients, img_size):
        # Create grid-based patch selection
        B, C, H, W = gradients.shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        
        # Calculate patch importance
        patch_grads = gradients.abs().amax(dim=1)  # Max across channels
        
        # Debug gradients
        # print(f"Gradient stats - Min: {patch_grads.min().item():.6f}, Max: {patch_grads.max().item():.6f}, Mean: {patch_grads.mean().item():.6f}")
        
        # If gradients are too small, add small constant to ensure non-zero
        if patch_grads.max().item() < 1e-6:
            print("Warning: Very small gradients detected, adding small constant")
            patch_grads = patch_grads + 1e-6
        
        patch_grads = patch_grads.unfold(1, self.patch_size, self.patch_size)
        patch_grads = patch_grads.unfold(2, self.patch_size, self.patch_size)
        patch_scores = patch_grads.mean(dim=(2, 3))  # [B, grid_h, grid_w]
        
        # Force selection of at least some patches even if gradients are small
        top_scores, top_indices = torch.topk(
            patch_scores.view(B, -1), min(self.num_patches, grid_h * grid_w), dim=1
        )
        
        # Create binary mask
        mask = torch.zeros((B, 1, H, W), device=gradients.device)
        for i in range(B):
            indices = top_indices[i]
            patches_filled = 0
            for idx in indices:
                h_idx = idx // grid_w
                w_idx = idx % grid_w
                h_start = h_idx * self.patch_size
                w_start = w_idx * self.patch_size
                
                # Ensure we don't go out of bounds
                if h_start + self.patch_size <= H and w_start + self.patch_size <= W:
                    mask[i, :, h_start:h_start+self.patch_size, 
                             w_start:w_start+self.patch_size] = 1
                    patches_filled += 1
            
            # print(f"Created mask for image {i}: {patches_filled} patches filled")
        
        return mask
    

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        def svm_boundary_loss(clip_embedding):
            # Distance to decision boundary (negative = wrong side)
            distance = torch.matmul(clip_embedding, self.svm_weights) + self.svm_bias
            # Loss is higher when distance is positive (correct classification)
            return -distance  # Maximize to cross boundary
            
        adv_images = images.clone().detach()
        
        # Initialize delta tensor to track perturbations separately
        delta = torch.zeros_like(images)
        
        if self.random_start:
            # Initialize perturbations only in patches to be selected
            random_noise = torch.randn_like(images) * self.eps
            # We'll apply this noise after selecting patches
        
        # Store patch mask for visualization
        final_patch_mask = None
        
        for _ in range(self.steps):
            # Apply current perturbations to selected patches
            curr_adv_images = images.clone().detach()
            if final_patch_mask is not None:
                curr_adv_images = images + delta * final_patch_mask
            
            curr_adv_images.requires_grad = True
            outputs = self.get_logits(curr_adv_images)
            
            loss = svm_boundary_loss(outputs)
# Convert loss to scalar by taking mean across batch
            loss = loss.mean()  # This fixes the gradient computation error
            print(f"Loss: {loss.item():.6f}")
            grad = torch.autograd.grad(loss, curr_adv_images, 
                                      retain_graph=False,
                                      create_graph=False)[0]
            
            # Create patch mask based on gradients
            patch_mask = self.create_patch_mask(grad, images.shape[-2:])
            final_patch_mask = patch_mask  # Save the mask for visualization
            
            # Apply initial random noise if this is the first iteration
            if _ == 0 and self.random_start:
                delta = random_noise * patch_mask
            
            # Update perturbations within the patches
            delta = delta - self.alpha * grad.sign() * patch_mask
            delta = torch.clamp(delta, min=-self.eps, max=self.eps)
            
            # Create adversarial image: original image with perturbations only in patches
            adv_images = torch.clamp(images + delta * patch_mask, 0, 1)
        
        # Create visualization with boxes around selected patches
        vis_images = images.clone()  # Start with original images
                
        # Draw MUCH BRIGHTER red boxes around the selected patches
        B, C, H, W = images.shape
        box_thickness = 2
                
        for i in range(B):
            # Find the patch boundaries from the mask
            mask = final_patch_mask[i, 0]
            
            # Count selected patches for debugging
            selected_count = 0
            
            # For each row and column
            for h in range(0, H - box_thickness, self.patch_size):
                for w in range(0, W - box_thickness, self.patch_size):
                    # Check the whole patch area instead of just the center point
                    patch_area = mask[h:h+self.patch_size, w:w+self.patch_size]
                    if patch_area.sum() > 0:  # If any pixel in patch is selected
                        selected_count += 1
                        
                        # Draw BRIGHT red box (R=1, G=0, B=0)
                        # Top and bottom edges
                        vis_images[i, 0, h:h+self.patch_size, w:w+box_thickness] = 1.0  # R
                        vis_images[i, 1, h:h+self.patch_size, w:w+box_thickness] = 0.0  # G
                        vis_images[i, 2, h:h+self.patch_size, w:w+box_thickness] = 0.0  # B
                        
                        vis_images[i, 0, h:h+self.patch_size, w+self.patch_size-box_thickness:w+self.patch_size] = 1.0
                        vis_images[i, 1, h:h+self.patch_size, w+self.patch_size-box_thickness:w+self.patch_size] = 0.0
                        vis_images[i, 2, h:h+self.patch_size, w+self.patch_size-box_thickness:w+self.patch_size] = 0.0
                        
                        # Left and right edges
                        vis_images[i, 0, h:h+box_thickness, w:w+self.patch_size] = 1.0
                        vis_images[i, 1, h:h+box_thickness, w:w+self.patch_size] = 0.0
                        vis_images[i, 2, h:h+box_thickness, w:w+self.patch_size] = 0.0
                        
                        vis_images[i, 0, h+self.patch_size-box_thickness:h+self.patch_size, w:w+self.patch_size] = 1.0
                        vis_images[i, 1, h+self.patch_size-box_thickness:h+self.patch_size, w:w+self.patch_size] = 0.0
                        vis_images[i, 2, h+self.patch_size-box_thickness:h+self.patch_size, w:w+self.patch_size] = 0.0
            
            print(f"Image {i}: Selected {selected_count} patches out of {self.num_patches} requested")

        return adv_images, vis_images
