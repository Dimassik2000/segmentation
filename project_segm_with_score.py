from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import requests

# Load the image from a URL
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Create a panoptic segmentation pipeline
panoptic_segmentation = pipeline("image-segmentation", model="facebook/mask2former-swin-large-cityscapes-panoptic")

# Perform segmentation on the input image
results = panoptic_segmentation(image)

# Path to save the road mask with probability overlay
output_path = "/misc/home6/s0185/segm_models/road_mask_with_score.png"

# Initialize variables to store road mask and its score
road_mask = None
road_score = None

# Print detected objects with their probabilities
print("Detected objects:")
for result in results:
    label = result['label']  # Object label
    score = result['score']  # Detection confidence
    print(f"Object: {label}, Probability: {score:.6f}")
    
    # Check if the detected object is 'road'
    if label == 'road':
        road_mask = result['mask']  # Save the mask for the road
        road_score = score  # Save the confidence score for the road
        break  # Stop iterating once the road is found

# If a road mask is found, add the probability overlay
if road_mask:
    # Convert the mask to RGBA format
    road_mask = road_mask.convert("RGBA")
    draw = ImageDraw.Draw(road_mask)
    
    # Text to display the road probability
    text = f"Road: {road_score:.2%}"
    font = ImageFont.load_default()  # Use default font
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]  # Get the text dimensions
    text_position = (10, 10)  # Position of the text (top-left corner)
    
    # Add a semi-transparent background for the text
    draw.rectangle(
        [text_position, (text_position[0] + text_width + 10, text_position[1] + text_height + 5)],
        fill=(255, 255, 255, 128)  # White background with transparency
    )
    
    # Draw the text on the image
    draw.text((text_position[0] + 5, text_position[1]), text, fill=(0, 0, 0, 255), font=font)
    
    # Save the road mask with the overlay
    road_mask.save(output_path)
    print(f"Road mask saved to: {output_path}")
else:
    # If no road mask is found, print a message
    print("Road mask not found.")
