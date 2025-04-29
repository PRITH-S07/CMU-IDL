from torchattacks import PGD
from models.discriminator import CLIPSVMDiscriminator
import torchvision
from torchvision.transforms import transforms
import torch

IMAGE_PATH = "images/dalle/dalle_1388.jpg"
# IMAGE_PATH = "./images/dalle_1388.jpg"


class CLIPPGDAttack(PGD):
    def __init__(self, model, svm, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__(model, eps, alpha, steps, random_start)
        # SVM weights will be set if available
        self.svm_weights = torch.FloatTensor(svm.coef_[0])
        self.svm_bias = torch.tensor(svm.intercept_[0])
  
    def get_logits(self, inputs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)

        # Get image features from the vision model
        vision_outputs = self.model.vision_model(inputs)
        image_features = vision_outputs.last_hidden_state[:, 0, :]

        return image_features
    
    def svm_boundary_loss(self, clip_embedding):
        # Distance to decision boundary (negative = wrong side)
        if self.svm_weights is None or self.svm_bias is None:
            raise ValueError("SVM weights and bias not set. Call set_svm_params() first.")
        
        distance = torch.matmul(clip_embedding, self.svm_weights) + self.svm_bias
        # Loss is higher when distance is positive (correct classification)
        return -distance  # Maximize to cross boundary
        
    def forward(self, images, labels):
        """
        Override forward method to use custom loss function
        """
        images = images.clone().detach().to(self.device)
        
        adv_images = images.clone().detach()
        
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            
            # Forward pass
            outputs = self.get_logits(adv_images)
            
            # Calculate loss using our custom SVM boundary loss
            loss = self.svm_boundary_loss(outputs).mean()
            print(f"Loss: {loss.item():.6f}")
            
            # Backward pass
            grad = torch.autograd.grad(loss, adv_images,
                                      retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    

def preprocess_image(image_path, resize_transform):
    # Load the image to torch
    image = torchvision.io.read_image(image_path)
    image = resize_transform(image)
    image = image.float() / 255.0  # Normalize to [0, 1]
    image = resize_transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image
    return image


if __name__ == "__main__":
    # Load the model
    model = CLIPSVMDiscriminator()
    discriminator = model.model
    attack = CLIPPGDAttack(
        discriminator, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True
    )

    # Set SVM parameters if available
    if model.svm_trained:
        attack.set_svm_params(model.svm)

    # Load the image to torch
    resize_transform = transforms.Resize((224, 224))
    image = preprocess_image(IMAGE_PATH, resize_transform)

    target_label = torch.tensor([0]).to(model.device)

    # Generate adversarial example
    adversarial_image = attack(image, target_label)

    # Save the adversarial image
    adv_temp_path = "adversarial_image_dalle_1388.png"
    torchvision.utils.save_image(adversarial_image.squeeze(0), adv_temp_path)

    og_image_embeddings = discriminator.run_clip(IMAGE_PATH).reshape(1, -1)
    adv_image_embeddings = discriminator.run_clip(adv_temp_path).reshape(1, -1)

    if discriminator.svm_trained:
        og_pred, og_prob = discriminator.predict_from_embeddings(og_image_embeddings)
        adv_pred, adv_prob = discriminator.predict_from_embeddings(adv_image_embeddings)

        print(f"Original prediction: {og_pred}")
        print(f"New prediction: {adv_pred}")
