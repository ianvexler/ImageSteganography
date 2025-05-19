# Image-to-Image Steganography for Watermark Creation

This project implements an image watermarking tool using image-to-image steganography. Leverages SIFT keypoints and Least Significant Bit (LSB) embedding. It includes a web interface using Flask for embedding, verifying, and detecting tampering in images.

## Requirements

- Python 3.8 or above
- The following Python libraries:
  - flask
  - opencv-python
  - numpy

## How to Install and Run

1. Make sure Python is installed on your system.

2. Install the required libraries by running the following command in the terminal:

   pip install flask opencv-python numpy

4. Run the Flask app from the root of the project:

   python app.py

5. Open your browser and visit:

   http://127.0.0.1:5000

## Notes

- Example images are located in the `static/images/` folder.
- Images were sourced from [Pixabay](https://pixabay.com/).