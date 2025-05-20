from flask import Flask, render_template, request, send_file, jsonify
from watermark import Watermark
import cv2
import numpy as np
import io
import base64

app = Flask(__name__)
wm = Watermark()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed():
    carrier = request.files.get('carrier')
    watermark_img = request.files.get('watermark')

    try:
        carrier_img = __decode_file(carrier)
        watermark_arr = __decode_file(watermark_img)

        result_img = wm.embed(carrier_img, watermark_arr)
        _, buffer = cv2.imencode('.png', result_img)
        img = base64.b64encode(buffer).decode('utf-8')

        response = {
            "result": True,
            "status": "success",
            "message": "Watermark embedded",
            "image": img,
            "image_name": 'embedded',
        }

    except Exception as e:
        print("Error during embeding:", e)
        response = {
            "result": False,
            "status": "danger",
            "message": "Error during embeding"
        }
    
    return jsonify(response)

@app.route('/recover', methods=['POST'])
def recover():
    image = request.files.get('image')
    watermark_img = request.files.get('watermark')

    try:
        image_arr = __decode_file(image)
        watermark_arr = __decode_file(watermark_img)

        result = wm.recover(image_arr, watermark_arr)

        response = {
            "result": result,
            "status": "success" if result else "danger",
            "message": f"Authenticity Verified: {'Yes' if result else 'No'}"
        }

    except Exception as e:
        print("Error during watermark recovery:", e)
        response = {
            "result": False,
            "status": "danger",
            "message": "Error during watermark recovery"
        }
    
    return jsonify(response)

@app.route('/tamper', methods=['POST'])
def tamper():
    image = request.files.get('image')
    watermark_img = request.files.get('watermark')

    try:
        image_arr = __decode_file(image)
        watermark_arr = __decode_file(watermark_img)

        verified, avg_similarity, result_img = wm.tampered(image_arr, watermark_arr)

        result = not (verified == 1.0)

        if result:
            message = f"Tampering Detected: {verified * 100:.1f}% of keypoints verified"

            if avg_similarity > 0.8:
                suggestion = "Small watermark mismatch."
            elif avg_similarity > 0.6:
                suggestion = "Moderate watermark mismatch."
            else:
                suggestion = "Significant watermark mismatch."

            message = f"{message}<br> Average Keypoint Similarity: {round(avg_similarity * 100, 2)}% <br> {suggestion}"

        else:
            message = "No Tampering Detected"

        _, buffer = cv2.imencode('.png', result_img)
        img = base64.b64encode(buffer).decode('utf-8')

        response = {
            "result": result,
            "status": "warning" if result else "secondary",
            "message": message,
            "image": img,
            "image_name": 'tampered'
        }
    
    except Exception as e:
        print("Error during tampering detection:", e)
        response = {
            "result": False,
            "status": "danger",
            "message": "Error during tampering detection"
        }

    return jsonify(response)

def __decode_file(file_obj):
    file_bytes = np.frombuffer(file_obj.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

if __name__ == '__main__':
    app.run(debug=True)
