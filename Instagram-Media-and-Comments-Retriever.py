import instaloader
import csv
import torch
import time
from PIL import Image
import cv2
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration

#function to get Instagram comments and caption
def get_ig_comments(post_url, username, image_path):
    L = instaloader.Instaloader()
    # load insta session 
    L.load_session_from_file(username, f"/Users/rubinaalmas/.config/instaloader/session-{username}")
    
    shortcode = post_url.split("/")[-2]
    post = instaloader.Post.from_shortcode(L.context, shortcode)

    #fetch the post caption
    caption = post.caption if post.caption else "No caption"
    
    comments_data = []
    for count, comment in enumerate(post.get_comments()):
        comments_data.append((comment.owner.username, comment.text, caption))

        #stop if we reach 1000 comments
        if len(comments_data) >= 1000:
            break
        
        #pause after every 100 comments
        if (count + 1) % 100 == 0:
            time.sleep(60)

    #OCR and BLIP text for the associated image
    ocr_text = process_image_for_description(image_path)
    blip_text = process_image_for_blip_description(image_path)

    return comments_data, ocr_text, blip_text

#function to extract OCR text using pytesseract
def process_image_for_description(image_path):
    """Extract and clean text from image using Tesseract OCR."""
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Unable to load image."

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(thresh)

    custom_config = r'--oem 3 --psm 6'
    raw_text = pytesseract.image_to_string(inverted, config=custom_config)

    return raw_text if raw_text.strip() else "No readable text found in the image."

#function to generate BLIP description for image
def process_image_for_blip_description(image_path):
    """Generate a caption for the image using the BLIP model."""
    try:
        image = Image.open(image_path)
    except Exception as e:
        return f"Error: Unable to load image. {str(e)}"

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = caption_model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

#initialize BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

post_url = "URL"
username = "USERNAME OF INSTAGRAM"
image_path = "DOWNLOAD IMAGE FROM INSTA POST AND PUT THE PATH HERE"  # Replace with your image path

comments_data, ocr_text, blip_text = get_ig_comments(post_url, username, image_path)

output_file = "Name your output file here"
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["OCR_Text", "Username", "Comment", "Caption", "BLIP_Text"])
    for user, comment, caption in comments_data:
        writer.writerow([ocr_text, user, comment, caption, blip_text])

print(f"Data saved to {output_file}")
