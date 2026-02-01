# Image-to-Image Steganography for Watermark Creation

This project implements an image watermarking tool using image-to-image steganography. It leverages SIFT keypoints and Least Significant Bit (LSB) embedding to imperceptibly embed watermarks into images. The system supports watermark embedding, verification, and tamper detection through a web interface built with Flask.

## Key Features

- SIFT-based keypoint selection for robust embedding locations  
- Adaptive watermark patches scaled and rotated per keypoint  
- LSB embedding in the blue channel for visual imperceptibility  
- Watermark recovery and tampering detection using Hamming similarity  
- Interactive Flask web interface for testing and evaluation  

---

## Results Summary

The watermarking approach was evaluated in terms of **visibility**, **recovery accuracy**, and **robustness to tampering**.

### Visual Imperceptibility

- The watermark introduces only minimal pixel-level changes by modifying the LSB of the blue channel.
- No visible difference can be observed between the original and watermarked images under normal viewing conditions.
- Absolute difference images confirm that changes are effectively imperceptible.

<table>
  <tr>
    <th>Original Image</th>
    <th>Watermarked Image</th>
  </tr>
  <tr>
    <td><img src="static/images/carrier.png" alt="Original" height="400"></td>
    <td><img src="static/images/embedded.png" alt="Watermarked" height="400"></td>
  </tr>
</table>

---

### Watermark Recovery

- In **unmodified images**, watermark recovery achieved a **100% success rate** across all tests.
- SIFT keypoints remained consistent after embedding, allowing accurate extraction of all watermark patches.
- This demonstrates strong reliability for authenticity verification under ideal conditions.

<p align="center">
  <img src="static/images/results/recovery1.png" alt="Successful Watermark Recovery" height="400">
</p>

---

### Tampering Detection

The system was tested against common image manipulations:

- **Rotation**
  - After a 90° rotation, approximately **48% of keypoints were fully verified**.
  - Average similarity across detected keypoints remained high (~93%), indicating partial robustness.

- **Resizing**
  - Reducing image resolution by 20% significantly degraded performance.
  - No watermark patches were perfectly recovered, and average similarity dropped to ~54%.

- **Cropping**
  - Cropped regions resulted in permanently lost keypoints, making some watermark patches unrecoverable.

Tampering results are visualised using colour-coded markers:
- **Green**: perfect match  
- **Orange**: partial match  
- **Red**: tampered or missing watermark  

| Tampering Detection After Rotation | Tampering Detection After Resizing |
|:----------------------------------:|:-----------------------------------:|
| ![Rotation](static/images/results/recovery2.png) | ![Resizing](static/images/results/recovery3.png) |

---

### Overall Assessment

- The system is highly effective for **imperceptible watermarking and verification**.
- It shows **good robustness to mild transformations**, such as rotation.
- Performance degrades under **aggressive scale changes or cropping**, reflecting the sensitivity of keypoint-based approaches.
- Hamming similarity enables meaningful tamper detection even when exact recovery fails.

---

## Requirements

- Python 3.8 or above
- Required libraries:
  - flask
  - opencv-python
  - numpy

---

## How to Install and Run

1. Ensure Python is installed on your system.

2. Install dependencies:

   ```bash
   pip install flask opencv-python numpy
