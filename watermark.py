import cv2
import os
import numpy as np

class Watermark:
    def __init__(self, n_keypoints=100, base_patch_size=7.0, max_patch_size=15):
        self.PATH = 'images'
        self.N_KEYPOINTS = n_keypoints
        
        self.BASE_PATCH_SIZE = base_patch_size
        self.MAX_PATCH_SIZE = max_patch_size

    def embed(self, img_filename, watermark_filename):
        print("Embedding watermark")
        img = self.__read_img(img_filename)
        watermark = self.__read_img(watermark_filename)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) 

        n_kps = self.__apply_sift(img_gray)
        adjusted_watermark = self.__adjust_watermark(watermark)

        count = 1
        for kp in n_kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])

            transformed_wm = self.__apply_transform(adjusted_watermark, kp)

            # TODO: Remove
            # print(f"---")
            # print(f"KP {count}: ({x}, {y}) | Angle: KP: {round(kp.angle, 1)}")
            # print(transformed_wm)
            # print(f"---")

            h_patch, w_patch = transformed_wm.shape
            x_diff = w_patch // 2
            y_diff = h_patch // 2

            submatrix = img[
                (y - y_diff):(y + y_diff + 1), 
                (x - x_diff):(x + x_diff + 1), 
                0
            ]
            if submatrix.shape[:2] == transformed_wm.shape:
                submatrix &= 254
                submatrix |= transformed_wm.astype(np.uint8)

            count += 1

        self.__write_img('embeded-img.png', img)
        img_saved = cv2.imread("images/embeded-img.png", cv2.IMREAD_UNCHANGED)
        return img

    def recover(self, img_filename, watermark_filename):
        print("Recovering watermark")
        img = self.__read_img(img_filename)
        watermark = self.__read_img(watermark_filename)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps = self.__apply_sift(img_gray)
        
        adjusted_watermark = self.__adjust_watermark(watermark)

        verified = True

        count = 1
        for kp in kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
        
            patch = self.__get_kp_patch(img, kp)
            transformed_wm = self.__apply_transform(adjusted_watermark, kp)

            match = np.sum(patch == transformed_wm)
            match_ratio = match / transformed_wm.size

            if match_ratio < 1.0:
                verified = False

                # TODO: Remove
                print(f"KP {count}: ({x}, {y}) | Angle: KP: {round(kp.angle, 1)}")

                print(patch)
                print(transformed_wm)
                break
            count += 1
        return verified

    def __get_kp_patch(self, img, kp):
        scale, patch_size = self.__scale_kp(kp)
        img_lsb = img[..., 0] & 1

        x, y = int(kp.pt[0]), int(kp.pt[1])
        half_size = patch_size // 2

        patch = img_lsb[
            max(y - half_size, 0):y + half_size + 1,
            max(x - half_size, 0):x + half_size + 1
        ]
        return patch

    def tampered(self, img_filename, watermark_filename):
        print("Checking for tampering")

        img = self.__read_img(img_filename)
        watermark = self.__read_img(watermark_filename)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps = self.__apply_sift(img_gray)

        adjusted_watermark = self.__adjust_watermark(watermark)

        verified, tampered = [], []

        for kp in kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])

            patch = self.__get_kp_patch(img, kp)
            transformed_wm = self.__apply_transform(adjusted_watermark, kp)
   
            match = np.sum(patch == transformed_wm)
            match_ratio = match / adjusted_watermark.size

            if match_ratio == 1.0:
                verified.append(kp)
            else:
                tampered.append(kp)

        total = len(verified) + len(tampered)
        verified_ratio = len(verified) / total if total > 0 else 0

        print(f"Verified keypoints: {len(verified)} / {total} ({verified_ratio:.2%})")
        
        tampered_img = cv2.drawKeypoints(img, tampered, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.__write_img('tampered-test-img.png', tampered_img)
        
        return verified_ratio < 1.0

    def __apply_transform(self, img, kp, inverse=False):
        scale, patch_size = self.__scale_kp(kp)

        img = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
        center = (patch_size // 2, patch_size // 2)
        
        angle = round(kp.angle, 1)
        angle = -angle if inverse else angle


        M = cv2.getRotationMatrix2D(center, angle, scale)

        output_center = (patch_size // 2, patch_size // 2)
        M[0, 2] += output_center[0] - center[0]
        M[1, 2] += output_center[1] - center[1]

        transformed_patch = cv2.warpAffine(img, M, (patch_size, patch_size), flags=cv2.INTER_NEAREST)
        transformed_patch = (transformed_patch * 255).astype(np.uint8)
        _, transformed_patch = cv2.threshold(transformed_patch, 127, 1, cv2.THRESH_BINARY)

        return transformed_patch

    # TODO: Maybe change this
    def __apply_sift(self, img_gray):
        sift = cv2.SIFT_create()
        kps = sift.detect(img_gray, None)

        h, w = img_gray.shape[:2]    
        pad = self.MAX_PATCH_SIZE // 2

        # Sort by strongest response
        kps = sorted(kps, key=lambda kp: -kp.response)

        valid_kps = []
        min_distance = self.MAX_PATCH_SIZE  # You can adjust this value if needed

        for kp in kps:
            # Get dynamic patch size for each keypoint
            scale, patch_size = self.__scale_kp(kp)
            patch_half_size = patch_size // 2

            x, y = kp.pt
            x -= pad
            y -= pad

            if not (0 <= x < w and 0 <= y < h):
                continue

            too_close = False
            for akp in valid_kps:
                # Compute distance between keypoints based on patch size
                dx = kp.pt[0] - akp.pt[0]
                dy = kp.pt[1] - akp.pt[1]

                # Adjust the distance threshold based on the patch sizes
                if dx * dx + dy * dy < (patch_half_size * 2) ** 2:  # Distance threshold
                    too_close = True
                    break

            if not too_close:
                valid_kps.append(kp)

            if len(valid_kps) == self.N_KEYPOINTS:
                break

        return valid_kps

    def __scale_kp(self, kp):
        scale = kp.size / self.BASE_PATCH_SIZE
        # scale = max(kp.size, self.BASE_PATCH_SIZE) / self.BASE_PATCH_SIZE
        
        patch_size = int(np.ceil(self.BASE_PATCH_SIZE * scale))
        patch_size = min(patch_size, self.MAX_PATCH_SIZE)
        patch_size = max(patch_size, int(self.BASE_PATCH_SIZE))

        if patch_size % 2 == 0:
            patch_size += 1

        return scale, patch_size

    def __adjust_watermark(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        _, img_binary = cv2.threshold(img_gray, 127, 1, cv2.THRESH_BINARY)
        return img_binary

    def __read_img(self, filename):
        img_path = self.__get_img_path(filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image from {img_path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img

    def __write_img(self, filename, img):
        img_path = self.__get_img_path(filename)
        cv2.imwrite(f'{img_path}', img)
        print(f"Watermarked image saved ({img_path})")

    def __get_img_path(self, filename):
        os.makedirs(self.PATH, exist_ok=True)
        return os.path.join(self.PATH, filename)

if __name__ == '__main__':
    watermark = Watermark()

    watermark.embed('img_rotated.png', 'watermark.png')
    print("")

    result = watermark.recover('embeded-img.png', 'watermark.png')
    print(f"Recovered: {result}\n")

    result = watermark.tampered('embeded-img.png', 'watermark.png')
    print(f"Tampered: {result}\n")
