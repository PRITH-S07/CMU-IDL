from PIL import Image
import os
from rembg import remove
import numpy as np
import cv2
import urllib


# Set the input and output directories
INPUT_DIR = "images/real_raw"
OUTPUT_DIR = "images/real"

CASCADE_FILE = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_FILE)


def crop_and_scale_image(input_path, output_path):
    img = Image.open(input_path)
    np_img = np.array(img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    # Face detection first

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        # Use largest face as focal point
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        x_center = x + w // 2
        y_center = y + h // 2
        print(f"Facial Crop of {input_path} to {output_path}")
    else:
        # # Fallback to subject detection
        # mask = remove(img).convert("L")
        # mask_data = np.array(mask)
        # y_nonzero, x_nonzero = np.nonzero(mask_data)

        # if len(y_nonzero) > 0 and len(x_nonzero) > 0:
        #     x_center = (x_nonzero.min() + x_nonzero.max()) // 2
        #     y_center = (y_nonzero.min() + y_nonzero.max()) // 2
        #     print(f"Focal Crop of {input_path} to {output_path}")
        # else:
        # Final fallback to image center
        x_center, y_center = img.width // 2, img.height // 2
        print(f"Center Crop of {input_path} to {output_path}")

    # Smart cropping
    # Revised smart cropping without black bars
    size = min(img.width, img.height)

    # Calculate initial crop coordinates
    left = x_center - size // 2
    top = y_center - size // 2
    right = left + size
    bottom = top + size

    # Adjust boundaries without losing size
    if left < 0:
        right -= left  # Add negative left to right
        left = 0
    elif right > img.width:
        left -= right - img.width
        right = img.width

    if top < 0:
        bottom -= top  # Add negative top to bottom
        top = 0
    elif bottom > img.height:
        top -= bottom - img.height
        bottom = img.height

    # Final clamp to ensure valid coordinates
    left = max(0, left)
    right = min(img.width, right)
    top = max(0, top)
    bottom = min(img.height, bottom)

    cropped = img.crop((left, top, right, bottom))
    resized = cropped.resize((512, 512), Image.LANCZOS)
    resized.save(output_path)


def downscale_1024_to_512(input_path, output_path):
    if input_path.lower().endswith(".jpg"):
        with Image.open(input_path) as img:
            # Check if the image is 1024x1024
            if img.size == (1024, 1024):
                # Resize the image to 512x512
                resized_img = img.resize((512, 512), Image.LANCZOS)

                # Save the resized image
                output_path = os.path.join(OUTPUT_DIR, filename)
                resized_img.save(output_path)
                print(f"Resized {input_path} to 512x512")
            else:
                print(f"Skipped {input_path} - not 1024x1024")


if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(INPUT_DIR):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.isfile(output_path):
            print(f"Image {output_path} already exists")
            continue
        crop_and_scale_image(input_path, output_path)
    print("Normalization complete!")
