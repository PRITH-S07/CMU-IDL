from adversarial_models.pgd import PGD
from models.discriminator import CLIPSVMDiscriminator
import torchvision
import torch

IMAGE_PATH = "images/dalle/dalle_148.jpg"

if __name__ == "__main__":
    # Load the model
    model = CLIPSVMDiscriminator()
    clip = model.model
    attack = PGD(clip, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True)

    # Load the image to torch
    image = torchvision.io.read_image(IMAGE_PATH)
    image = image.float() / 255.0  # Normalize to [0, 1]
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(model.device)

    # Generate adversarial example
    adversarial_image = attack(image, torch.tensor([0]).to(model.device))
    adversarial_image = adversarial_image.squeeze(0)  # Remove batch dimension
    adversarial_image = (adversarial_image * 255).byte()  # Convert back to [0, 255]

    # Save the adversarial image
    torchvision.utils.save_image(adversarial_image, "adversarial_image.png")