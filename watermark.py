import cv2
import os
import numpy as np

class Watermark:
    def __init__(
        self, 
        n_keypoints=100, 
        watermark_shape=(7,7), 
        rotation_threshold=0.8, 
        cropping_threshold=0.9
    ):
        self.PATH = 'images'
        self.N_KEYPOINTS = n_keypoints
        self.WATERMARK_SHAPE = watermark_shape

        self.ROTATION_THRESHOLD = 0.8
        self.CROPPING_THRESHOLD = rotation_threshold
    
    # Embed watermark on image
    def embed(self, img_filename, watermark_filename):
        img = self.__read_img(img_filename)
        watermark = self.__read_img(watermark_filename)

        # Convert carrier img to grayscale
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY) 
        n_kps = self.__apply_sift(img_gray)

        # Adjust watermark
        adjusted_watermark = self.__adjust_watermark(watermark)

        # Allows to change the watermark size
        x_diff = self.WATERMARK_SHAPE[0] // 2
        y_diff = self.WATERMARK_SHAPE[1] // 2

        for kp in n_kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            angle = kp.angle

            if x >= x_diff and y >= y_diff and x < img.shape[1] - x_diff and y < img.shape[0] - y_diff:
                # Rotates the watermark to match the kp angle
                rotated_watermark = self.__rotate_image(adjusted_watermark, angle)

                # Ensure binary
                rotated_watermark = (rotated_watermark > 0.5).astype(np.uint8)
                
                # Iterate through all colour channels B, G, R
                for c in range(3):
                    submatrix = img[
                        (y - y_diff):(y + y_diff + 1), 
                        (x - x_diff):(x + x_diff + 1), 
                        c
                    ]
                    submatrix &= 254
                    submatrix |= rotated_watermark.astype(np.uint8)

        img_embeded = self.__write_img('embeded-img', img)
        return img_embeded

    def recover(self, img_filename, watermark_filename):
        img = self.__read_img(img_filename)
        watermark = self.__read_img(watermark_filename)

        # Grayscale for SIFT
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps = self.__apply_sift(img_gray)

        # Adjust watermark to binary matrix (e.g., 3x3)
        adjusted_watermark = self.__adjust_watermark(watermark)

        x_diff = self.WATERMARK_SHAPE[0] // 2
        y_diff = self.WATERMARK_SHAPE[1] // 2

        verified = []
        tampered = []
        
        for kp in kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            angle = kp.angle


            if x >= x_diff and y >= y_diff and x < img.shape[1] - x_diff and y < img.shape[0] - y_diff:
                found = False

                for c in range(3):  # B, G, R
                    submatrix = img[
                        (y - y_diff):(y + y_diff + 1),
                        (x - x_diff):(x + x_diff + 1),
                        c
                    ]

                    lsb = submatrix & 1
                    
                    rotated_patch = self.__rotate_image(lsb, -angle)

                    # Ensure binary
                    rotated_patch_binary = (rotated_patch > 0.5).astype(np.uint8)

                    # Compare to original watermark
                    if np.array_equal(rotated_patch_binary, adjusted_watermark):
                        return True

    def tampered(self, img_filename, watermark_filename, threshold=0.7):
        img = self.__read_img(img_filename)
        watermark = self.__read_img(watermark_filename)

        # Grayscale for SIFT
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps = self.__apply_sift(img_gray)

        # Adjust watermark to binary matrix (e.g., 3x3)
        adjusted_watermark = self.__adjust_watermark(watermark)

        x_diff = self.WATERMARK_SHAPE[0] // 2
        y_diff = self.WATERMARK_SHAPE[1] // 2

        verified = []
        tampered = []

        # To detect cropping
        if len(kps) < (self.N_KEYPOINTS * self.CROPPING_THRESHOLD):
            missign_kps = self.N_KEYPOINTS - len(kps)
            
            print("Cropping detected")
            print(f"Missing kps: {missign_kps}")

        for kp in kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            angle = kp.angle
            scale = kp.size

            if x >= x_diff and y >= y_diff and x < img.shape[1] - x_diff and y < img.shape[0] - y_diff:
                found = False
               
                # B, G, R
                for c in range(3):
                    submatrix = img[
                        (y - y_diff):(y + y_diff + 1),
                        (x - x_diff):(x + x_diff + 1),
                        c
                    ]

                    lsb = submatrix & 1

                    # Rotates patch to be flat 
                    rotated_patch = self.__rotate_image(lsb, -angle)

                    # Ensure binary
                    rotated_patch_binary = (rotated_patch > 0.5).astype(np.uint8)

                    # Compares possible watermark to its original binary version
                    # No need to check other channels if equal                    
                    match = np.sum(rotated_patch_binary == adjusted_watermark)
                    match_ratio = match / adjusted_watermark.size

                    # Match has to match threshold
                    if match_ratio >= self.ROTATION_THRESHOLD:
                        found = True
                        break

                if found:
                    verified.append((x, y))
                else:
                    tampered.append((x, y))        

        total = len(verified) + len(tampered)
        verified_ratio = len(verified) / total

        print(f"Verified keypoints: {len(verified)} / {total} ({verified_ratio:.2%})")
        
        if verified_ratio < threshold:
            print("Tampering detected")
            return True
        else:
            print("Image is authenticated")
            return False
    
    # Rotates an image given an angle
    def __rotate_image(self, img, angle):
        h, w = img.shape
        center = (w // 2, h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # TODO: Remove border and flags?
        rotated = cv2.warpAffine(
            img,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return rotated


    # Apply SIFT
    def __apply_sift(self, img_gray):
        sift = cv2.SIFT_create()
        kps = sift.detect(img_gray, None)
        n_kps = kps[:self.N_KEYPOINTS]

        return n_kps

    # Adjust watermark by resizing and converting to binary
    def __adjust_watermark(self, img):
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
        img_resized = cv2.resize(img_gray, self.WATERMARK_SHAPE, interpolation=cv2.INTER_AREA)
        _, img_binary = cv2.threshold(img_resized, 127, 1, cv2.THRESH_BINARY)
        
        return img_binary

    # Reads image and ensures it has 4 channels
    def __read_img(self, filename):
        img_path = self.__get_img_path(filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Could not read image from {input_path}")
        
        # Grayscale
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        # BGR
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        return img

    def __write_img(self, filename, img):
        img_path = self.__get_img_path(f"{filename}.png")
        cv2.imwrite(f'{img_path}', img)

        print(f"Watermarked image saved ({img_path})")

    def __get_img_path(self, filename):
        os.makedirs(self.PATH, exist_ok=True)
        file_path = os.path.join(self.PATH, filename)
        return file_path

if __name__=='__main__':
    watermark = Watermark()
    watermark.embed('test.jpg', 'watermark.png')
    watermark.tampered('embeded-img.png', 'watermark.png')