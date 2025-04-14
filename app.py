from flask import Flask, request, jsonify, send_file
import os
from uuid import uuid4
from processor import process_video, process_image  # Make sure both functions exist

app = Flask(__name__)

# Video paths
VIDEO_UPLOAD_FOLDER = "input_videos"
VIDEO_OUTPUT_FOLDER = "output_videos"

# Image paths
IMAGE_UPLOAD_FOLDER = "input_images"
IMAGE_OUTPUT_FOLDER = "output_images"

# Server base URL (use your actual server IP if deploying)
BASE_URL = "http://127.0.0.1:5000"

# Ensure folders exist
os.makedirs(VIDEO_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_FOLDER, exist_ok=True)

# ----------------------------------------
# 🎥 Video Analyzer Endpoint
# ----------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files['video']
    input_filename = f"{uuid4()}.mp4"
    output_filename = f"processed_{uuid4()}.mp4"

    input_path = os.path.join(VIDEO_UPLOAD_FOLDER, input_filename)
    output_path = os.path.join(VIDEO_OUTPUT_FOLDER, output_filename)

    try:
        video_file.save(input_path)
        process_video(input_path, output_path)

        if not os.path.exists(output_path):
            return jsonify({"error": "Processing failed"}), 500

        video_url = f"{BASE_URL}/videos/{output_filename}"
        return jsonify({
            "success": True,
            "video_url": video_url,
            "message": "Video processed successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve processed video
@app.route('/videos/<filename>')
def get_video(filename):
    path = os.path.join(VIDEO_OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Video not found"}), 404
    return send_file(path, mimetype='video/mp4')


# ----------------------------------------
# 🖼️ Image Analyzer Endpoint
# ----------------------------------------
@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    input_filename = f"{uuid4()}.jpg"
    output_filename = f"processed_{uuid4()}.jpg"

    input_path = os.path.join(IMAGE_UPLOAD_FOLDER, input_filename)
    output_path = os.path.join(IMAGE_OUTPUT_FOLDER, output_filename)

    try:
        image_file.save(input_path)
        process_image(input_path, output_path)

        if not os.path.exists(output_path):
            return jsonify({"error": "Image processing failed"}), 500

        image_url = f"{BASE_URL}/images/{output_filename}"
        return jsonify({
            "success": True,
            "image_url": image_url,
            "message": "Image processed successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve processed image
@app.route('/images/<filename>')
def get_image(filename):
    path = os.path.join(IMAGE_OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Image not found"}), 404
    return send_file(path, mimetype='image/jpeg')


# ----------------------------------------
# 🟢 Run App
# ----------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
