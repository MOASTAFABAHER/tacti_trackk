# app.py
from flask import Flask, request, send_file
import os
from uuid import uuid4
from processor import process_video

app = Flask(__name__)

UPLOAD_FOLDER = "input_videos"
OUTPUT_FOLDER = "output_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'video' not in request.files:
        return {"error": "No video uploaded"}, 400
    
    video_file = request.files['video']
    input_filename = f"{uuid4()}.mp4"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    output_path = os.path.join(OUTPUT_FOLDER, input_filename)

    video_file.save(input_path)
    process_video(input_path, output_path)

    return send_file(output_path, mimetype='video/mp4')

if __name__ == "__main__":
    app.run(debug=True)
