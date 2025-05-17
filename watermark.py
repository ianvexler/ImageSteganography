import cv2
import os
import numpy as np

class Watermark:
    def __init__(self, n_keypoints=50, base_patch_size=7.0, max_patch_size=15):
        self.N_KEYPOINTS = n_keypoints
        
        self.BASE_PATCH_SIZE = base_patch_size
        self.MAX_PATCH_SIZE = max_patch_size

    def embed(self, img, watermark):
        print("Embedding watermark")

        img_gray = self.__img_to_grayscale(img)

        n_kps = self.__apply_sift(img_gray)
        adjusted_watermark = self.__adjust_watermark(watermark)

        count = 1
        for kp in n_kps:
            print(f"KP: {kp.pt}, angle: {kp.angle}, size: {kp.size}")
            x, y = int(kp.pt[0]), int(kp.pt[1])

            transformed_wm = self.__apply_transform(adjusted_watermark, kp)

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

        return img

    def recover(self, img, watermark):
        print("Recovering watermark")

        img_gray = self.__img_to_grayscale(img)
        kps = self.__apply_sift(img_gray)
        
        adjusted_watermark = self.__adjust_watermark(watermark)

        verified = True

        for kp in kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
        
            patch = self.__get_kp_patch(img, kp)
            transformed_wm = self.__apply_transform(adjusted_watermark, kp)

            if not np.array_equal(patch, transformed_wm):
                verified = False
                break

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

    def tampered(self, img, watermark):
        print("Checking for tampering")

        img_copy = img.copy()

        img_gray = self.__img_to_grayscale(img)
        kps = self.__apply_sift(img_gray)

        adjusted_watermark = self.__adjust_watermark(watermark)

        tampered = []

        for kp in kps:
            print(f"KP: {kp.pt}, angle: {kp.angle}, size: {kp.size}")
            x, y = int(kp.pt[0]), int(kp.pt[1])

            patch = self.__get_kp_patch(img, kp)
            transformed_wm = self.__apply_transform(adjusted_watermark, kp)

            if not np.array_equal(patch, transformed_wm):
                similarity = self.__calc_patch_similarity(patch, transformed_wm)

                tampered.append((kp, similarity))
                cv2.circle(img_copy, (x, y), radius=4, color=(0, 0, 255), thickness=2)

        img_tampered_keypoints = None

        if len(tampered):
            tampered_kps = [t[0] for t in tampered]

            img_tampered_keypoints = cv2.drawKeypoints(
                img, tampered_kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

            # Adds arrows to each tampered keypoint
            for kp in tampered_kps:
                pt = (int(kp.pt[0]), int(kp.pt[1]))
                
                start_pt = (pt[0] - 20, pt[1] - 20)
                cv2.arrowedLine(
                    img_tampered_keypoints, 
                    start_pt, 
                    pt, 
                    color=(255, 0, 0),
                    thickness=2,
                    tipLength=0.3
                )

        if len(tampered):
            avg_similarity = np.mean([sim for _, sim in tampered])
        else:
            avg_similarity = 1.0

        return ((len(kps) - len(tampered)) / len(kps)), avg_similarity, img_tampered_keypoints

    def __apply_transform(self, img, kp, inverse=False):
        scale, patch_size = self.__scale_kp(kp)

        img = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
        center = (patch_size // 2, patch_size // 2)
        
        angle = round(kp.angle, 0)
        angle = -angle if inverse else angle

        M = cv2.getRotationMatrix2D(center, angle, scale)

        output_center = (patch_size // 2, patch_size // 2)
        M[0, 2] += output_center[0] - center[0]
        M[1, 2] += output_center[1] - center[1]

        transformed_patch = cv2.warpAffine(img, M, (patch_size, patch_size), flags=cv2.INTER_NEAREST)
        transformed_patch = (transformed_patch * 255).astype(np.uint8)
        _, transformed_patch = cv2.threshold(transformed_patch, 127, 1, cv2.THRESH_BINARY)

        return transformed_patch

    def __apply_sift(self, img_gray):
        sift = cv2.SIFT_create()
        kps = sift.detect(img_gray, None)

        h, w = img_gray.shape[:2]    
        pad = self.MAX_PATCH_SIZE // 2

        # Sort by strongest response
        kps = sorted(kps, key=lambda kp: -kp.response)

        valid_kps = []
        min_distance = self.MAX_PATCH_SIZE

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

    def __calc_patch_similarity(self, patch, wm):
        hamming = np.sum(patch != wm)
        score = hamming / wm.size

        return score

    def __img_to_grayscale(self, img):
        if len(img.shape) == 2:
            return img
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
