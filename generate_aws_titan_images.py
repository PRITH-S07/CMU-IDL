import os
import json
import boto3
import base64
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import random

load_dotenv(override=True)

SESSION = boto3.session.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
BEDROCK = SESSION.client("bedrock-runtime", region_name="us-east-1")

ROOT = os.getcwd()
STYLES = ["impressionist", "expressionist"]
PROMPT_BASE = "Create an {style}-style painting which depicts the following scene: "

MODEL_ID = "amazon.titan-image-generator-v2:0"


def collect_images_prompt(caption_file="ArtCap.json"):
    prompt_base = PROMPT_BASE.format(style=random.choice(STYLES))
    with open(os.path.join(ROOT, caption_file), "r") as file:
        captions = json.load(file)
    prompt_dict = {i: prompt_base + ". ".join(caps) for i, caps in captions.items()}
    return prompt_dict


def generate_image(prompt):
    # Prepare the request body
    if len(prompt) > 512:
        prompt = prompt[:512]
    request_body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": prompt[:512]},
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "standard",
            "cfgScale": 8.0,
            "height": 512,
            "width": 512,
            "seed": 0,
        },
    }
    # Invoke the model
    response = BEDROCK.invoke_model(modelId=MODEL_ID, body=json.dumps(request_body))
    # Parse the response
    response_body = json.loads(response["body"].read())

    # Extract and decode the base64 image data
    image_data = base64.b64decode(response_body["images"][0])

    return image_data


##ADD API CALL RETURN IMAGE
def generate_and_save_images(
    image_prompt_dict,
    generator="titan",
    output_dir="generated_images",
    max_invocations=1,
):

    os.makedirs(os.path.join(ROOT, output_dir), exist_ok=True)
    flagged_images_path = os.path.join(output_dir, "flagged_images.txt")
    try:
        with open(flagged_images_path, "r") as f:
            flagged = set(line.strip() for line in f)
    except FileNotFoundError:
        flagged = set()

    num_invocations = 0
    for image, prompt in image_prompt_dict.items():
        output_file = f"{generator}_{image}"
        output_path = os.path.join(output_dir, output_file)
        try:
            if os.path.isfile(output_path):
                print(f"{image}\t| Image already exists")
                continue
            elif output_path in flagged:
                print(f"{image}\t| Prompt is in blacklist")
                continue
            else:
                num_invocations += 1
                image_data = generate_image(prompt)
                with open(output_path, "wb") as file:
                    file.write(image_data)
                print(f"{image}\t| Successfully created image")
        except ClientError as e:
            is_content_error = "blocked" in str(e)
            is_validation_exception = (
                e.response["Error"]["Code"] == "ValidationException"
            )
            if is_validation_exception and is_content_error:
                if output_path not in flagged:
                    with open(flagged_images_path, "a") as f:
                        f.write(f"{output_path}\n")
                    flagged.add(output_path)
                print(f"{image}\t| Inappropriate content detected by API")
            else:
                print(f"{image}\t| {e}")
        except Exception as e:
            print(f"{image}\t| {e}")

        if num_invocations >= max_invocations:
            break

    # return number of successfuly written images
    return len(os.listdir(os.path.join(ROOT, output_dir)))


if __name__ == "__main__":
    image_prompt_dict = collect_images_prompt()
    print(len(image_prompt_dict))
    success_count = generate_and_save_images(
        image_prompt_dict,
        generator="titan",
        output_dir="images/titan",
        max_invocations=1500,
    )
    print(f"Generated {success_count} images successfully")
