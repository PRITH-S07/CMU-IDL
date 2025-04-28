import os
import json
from huggingface_hub import InferenceClient
from PIL import Image

# Set working directory and base prompt text
ROOT = os.getcwd()
PROMPT_BASE = "Create an impressionist/expressionist-style painting which depicts the following scene: "


def collect_images_prompt(prompt_base, caption_file="ArtCap.json"):
    """
    Load captions from a JSON file and build a dictionary of prompts.
    The JSON is expected to be in the form:
        { "id1": ["caption part 1", "caption part 2", ...],
          "id2": ["caption part A", "caption part B", ...], ... }
    """
    with open(os.path.join(ROOT, caption_file), "r") as file:
        captions = json.load(file)
    # Concatenate the prompt base with the joined caption parts.
    prompt_dict = {
        i: prompt_base + " " + ". ".join(caps) for i, caps in captions.items()
    }
    return prompt_dict


import time
import requests
from huggingface_hub import InferenceClient

# Initialize the InferenceClient once, globally
client = InferenceClient(
    provider="hf-inference",
    api_key="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",
)


def generate_image(prompt, max_retries=3, delay=5):
    """
    Generate an image using the Hugging Face InferenceClient with retries on 503 errors.
    Returns a tuple: (PIL.Image or None, status_code).
    """
    for attempt in range(max_retries):
        try:
            image = client.text_to_image(
                prompt, model="stable-diffusion-v1-5/stable-diffusion-v1-5"
            )
            # If successful, return the image with status code 200
            return image, 200

        except requests.exceptions.HTTPError as e:
            # If we specifically get a 503, we can retry
            if e.response.status_code == 503:
                print(
                    f"503 error on attempt {attempt+1}. Retrying in {delay} seconds..."
                )
                time.sleep(delay)
            else:
                print(f"HTTPError (status code {e.response.status_code}): {e}")
                return None, e.response.status_code

        except Exception as e:
            # Handle any other exceptions
            print(f"Error generating image: {e}")
            return None, 1000

    # If all retries fail, return None
    return None, 503


def generate_and_save_images(
    image_prompt_dict, generator="stable_diff", output_dir="generated_images"
):
    """
    Iterate over the prompt dictionary, generate images, and save them.
    """
    os.makedirs(os.path.join(ROOT, output_dir), exist_ok=True)
    success_count = 0
    for image_id, prompt in image_prompt_dict.items():
        output_file = os.path.join(output_dir, f"{generator}_{image_id}")
        img, status_code = generate_image(prompt)
        if status_code == 200 and img is not None:
            # Save the generated PIL image to the output file.
            img.save(output_file)
            success_count += 1
            print(f"Saved image {image_id} to {output_file}")
        else:
            print(
                f"Error generating image for image id: {image_id} (status code: {status_code})"
            )
    return success_count


if __name__ == "__main__":
    # Collect image prompts from ArtCap.json using the provided prompt base.
    image_prompt_dict = collect_images_prompt(PROMPT_BASE)
    # Generate images and save them in the 'generated_images' directory.
    success_count = generate_and_save_images(image_prompt_dict)
    print(f"Successfully generated {success_count} images.")
    # Uncomment the following line if you expect a specific number of images (e.g., 3606)
    # assert(success_count == 3606)
