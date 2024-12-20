import os
import instaloader
import pytesseract
import cv2
from pydub import AudioSegment
import speech_recognition as sr
import pandas as pd

# function to download media and extract audio/image
def download_ig_post_with_audio(post_url, username, download_dir):
    L = instaloader.Instaloader()
    L.load_session_from_file(username, f"/Users/rubinaalmas/.config/instaloader/session-{username}")
    shortcode = post_url.split("/")[-2]
    post = instaloader.Post.from_shortcode(L.context, shortcode)

    #create a directory for downloads
    os.makedirs(download_dir, exist_ok=True)

    #download media (image and audio as video)
    media_path = None
    try:
        for resource in post.get_sidecar_nodes():
            is_video = resource.get("is_video", False)
            ext = ".mp4" if is_video else ".jpg"
            media_url = resource["video_url"] if is_video else resource["display_url"]
            media_path = os.path.join(download_dir, f"{shortcode}{ext}")
            L.download_pic(media_path, media_url, post.date_utc)
            break 
    except Exception as e:
        print(f"Error while downloading media: {e}")
        return None, "No caption"

    caption = post.caption if post.caption else "No caption"
    return media_path, caption


#function to process image for OCR text
def process_image_for_description(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Unable to load image."

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw_text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6')
    return raw_text.strip() if raw_text.strip() else "No readable text found in the image."

#function to extract and transcribe audio
def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        return "Sorry, I'm unable to reach the speech recognition service."

# function to convert video (mp4) to audio (wav)
def extract_audio_from_video(video_path, output_audio_path):
    audio = AudioSegment.from_file(video_path, format="mp4")
    audio.export(output_audio_path, format="wav")
    return output_audio_path

# function to fetch comments and usernames (Placeholder implementation)
def fetch_comments_and_usernames(post_url, username):
    # Use Instaloader or Instagram Graph API for this.
    # Placeholder data
    comments = ["This is a great post!", "Amazing content!", "Love this!"]
    usernames = ["user1", "user2", "user3"]
    return list(zip(usernames, comments))

# main Integration
def integrate_photo_and_audio(post_url, username, output_csv, download_dir="ig_media"):
    media_path, caption = download_ig_post_with_audio(post_url, username, download_dir)

    if media_path is None:
        print("No media was found for the given post URL.")
        return

    # process media
    ocr_text, audio_text = None, None
    try:
        if media_path.endswith(".jpg"):
            ocr_text = process_image_for_description(media_path)
        elif media_path.endswith(".mp4"):
            audio_path = os.path.join(download_dir, "extracted_audio.wav")
            extract_audio_from_video(media_path, audio_path)
            audio_text = audio_to_text(audio_path)
    except Exception as e:
        print(f"Error while processing media: {e}")

    # fetch comments and usernames
    comments_and_usernames = fetch_comments_and_usernames(post_url, username)

    # combine data
    data = []
    for user, comment in comments_and_usernames:
        data.append({
            "audio_from_text": audio_text,
            "ocr_text": ocr_text,
            "username": user,
            "comment": comment,
            "caption": caption
        })

    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Data integrated and saved to {output_csv}")

# Example Usage
post_url = "post url"
username = "username"
output_csv = "final_output.csv"

integrate_photo_and_audio(post_url, username, output_csv)
