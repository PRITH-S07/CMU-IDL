import os
import json
import requests

ROOT = os.getcwd()
PROMPT_BASE = "Create an impressionist/expressionist-style painting with bold colors, dynamic brushstrokes, and an evocative atmosphere. The image should depict the following scene: "



def collect_images_prompt(prompt_base, caption_file = "ArtCap.json"):
    with open(os.path.join(ROOT, caption_file), 'r') as file:
        captions = json.load(file)
    prompt_dict = {i : prompt_base + '. '.join(caps) for i, caps in captions.items()}
    return prompt_dict

def generate_image(prompt): ##TO DO -> create function for your API which takes in prompt and returns requests response
    response = None
    return response

##ADD API CALL RETURN IMAGE
def generate_and_save_images(image_prompt_dict, generator = "firefly", output_dir = "generated_images"):
    os.makedirs(os.path.join(ROOT, output_dir), exist_ok = True)
    for image, prompt in image_prompt_dict.items():        
        output_file = f"{generator}_{image}"
        
        ##COMPLETE THIS FUNCTION
        response = generate_image(prompt)
        
        #save image
        if response.status_code == 200:
            with open(output_file, "wb") as file:
                file.write(response.content)
        else:
            print(f"smh try again. status code: {response.status_code}")
    
    #return number of successfuly written images
    return len(os.listdir(os.path.join(ROOT, output_dir)))


image_prompt_dict = collect_images_prompt(PROMPT_BASE)
generate_and_save_images(image_prompt_dict)
