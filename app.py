from flask import Flask, render_template, request, send_file
from watermark import Watermark
import cv2
import numpy as np
import io

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

    result = wm.recover(image_arr, watermark_arr)
    return "Authenticity Verified: Yes" if result else "Authenticity Verified: No"


@app.route('/tamper', methods=['POST'])
def tamper():
    image = request.files.get('image')
    watermark_img = request.files.get('watermark')

    if not image or not watermark_img:
        return "Missing file(s).", 400

    image_arr = __decode_file(image)
    watermark_arr = __decode_file(watermark_img)

    result, _ = wm.tampered(image_arr, watermark_arr)
    return "Tampering Detected: Yes" if result else "Tampering Detected: No"

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
