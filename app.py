from flask import Flask, render_template, request, send_file, jsonify
from watermark import Watermark
import cv2
import numpy as np
import io
import traceback
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

    if not carrier or not watermark_img:
        return "Missing file(s).", 400

    carrier_img = __decode_file(carrier)
    watermark_arr = __decode_file(watermark_img)

    result_img = wm.embed(carrier_img, watermark_arr)

    _, buffer = cv2.imencode('.png', result_img)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/png', as_attachment=True, download_name='embedded.png')

@app.route('/recover', methods=['POST'])
def recover():
    image = request.files.get('image')
    watermark_img = request.files.get('watermark')

    if not image or not watermark_img:
        return "Missing file(s).", 400

    image_arr = __decode_file(image)
    watermark_arr = __decode_file(watermark_img)

    try:
        result = wm.recover(image_arr, watermark_arr)

        response = {
            "result": result,
            "status": "success" if result else "danger",
            "message": f"Authenticity Verified: {"Yes" if result else "No"}"
        }

        return jsonify(response)
    except Exception as e:
        print("Error during watermark recovery:", e)
        traceback.print_exc()
        return "Error whilst recovering"

@app.route('/tamper', methods=['POST'])
def tamper():
    image = request.files.get('image')
    watermark_img = request.files.get('watermark')

    image_arr = __decode_file(image)
    watermark_arr = __decode_file(watermark_img)

    try:
        verified, avg_similarity, output_img = wm.tampered(image_arr, watermark_arr)

        result = not (verified == 1.0)

        if result:
            message = f"Tampering Detected: {verified * 100:.1f}% of keypoints verified"

            if avg_similarity > 0.85:
                suggestion = "Minimal watermark mismatch."
            elif avg_similarity > 0.6:
                suggestion = "Moderate watermark mismatch."
            else:
                suggestion = "Significant watermark mismatch."

            message = f"{message}<br> Average Keypoint Similarity: {round(avg_similarity * 100, 2)}% <br> {suggestion}"

        else:
            message = "No Tampering Detected"

        response = {
            "result": result,
            "status": "warning" if result else "secondary",
            "message": message
        }
    except Exception as e:
        print("Error during watermark recovery:", e)
        traceback.print_exc()
        return "Error whilst detecting"

    _, buffer = cv2.imencode('.png', output_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    response["image"] = img_base64

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
