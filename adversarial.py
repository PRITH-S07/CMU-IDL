from adversarial_models.pgd import PGD
from models.discriminator import CLIPSVMDiscriminator
import torchvision
from torchvision.transforms import transforms
import torch

# IMAGE_PATH = "images/dalle/dalle_148.jpg"
IMAGE_PATH = "./images/dalle_1388.jpg"

class CLIPPGDAttack(PGD):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, random_start=True):
        super().__init__(model, eps, alpha, steps, random_start)
        
    def get_logits(self, inputs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        
        # Get image features from the vision model
        vision_outputs = self.model.vision_model(inputs)
        image_features = vision_outputs.last_hidden_state[:, 0, :]
        
        return image_features


def preprocess_image(image_path, resize_transform):
     # Load the image to torch
    image = torchvision.io.read_image(IMAGE_PATH)
    image = resize_transform(image)    
    image = image.float() / 255.0  # Normalize to [0, 1]
    image = resize_transform(image)   
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(model.device)
    return image
    
if __name__ == "__main__":
    # Load the model
    model = CLIPSVMDiscriminator()
    discriminator = model.model
    attack = CLIPPGDAttack(discriminator, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True)
    

    # Load the image to torch
    resize_transform = transforms.Resize((224, 224))
    image = preprocess_image(IMAGE_PATH, resize_transform)
    
    
    target_label = torch.tensor([0]).to(model.device)

    # Generate adversarial example
    adversarial_image = attack(image, target_label)

    # Save the adversarial image
    adv_temp_path =  "adversarial_image.png"
    torchvision.utils.save_image(adversarial_image.squeeze(0), adv_temp_path)
    
    
    og_image_embeddings = discriminator.run_clip(IMAGE_PATH).reshape(1,-1)
    adv_image_embeddings = discriminator.run_clip(adv_temp_path).reshape(1,-1)
        
    if discriminator.svm_trained:
        og_pred, og_prob = discriminator.predict_from_embeddings(og_image_embeddings)
        adv_pred, adv_prob = discriminator.predict_from_embeddings(adv_image_embeddings)
        
        print(f"Original prediction: {og_pred}")
        print(f"New prediction: {adv_pred}")