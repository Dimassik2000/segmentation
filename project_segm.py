from transformers import pipeline
from PIL import Image
import requests


url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg"
image = Image.open(requests.get(url, stream=True).raw)

semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
results = semantic_segmentation(image)

road_mask = None
for result in results:
    if result['label'] == 'road':
        road_mask = result['mask']
        break

if road_mask:
    road_mask.save("/misc/home6/s0185/segm_models/road_mask.png")
    print("Road mask saved as road_mask.png")
else:
    print("Road mask not found.")
