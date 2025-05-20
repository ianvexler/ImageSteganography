import cv2
import numpy as np

class Watermark:
    def __init__(self, n_keypoints=50, base_patch_size=7.0, max_patch_size=15):
        self.N_KEYPOINTS = n_keypoints
        self.BASE_PATCH_SIZE = base_patch_size
        self.MAX_PATCH_SIZE = max_patch_size

    def embed(self, img, watermark):
        """
        Embeds a binary watermark into an image by modifying the least significant bit (LSB)
        of the blue channel at keypoints detected via SIFT.
        
        Returns:
            - The image with embeded watermarks
        """

        # Identify kps on carrier image using sift
        n_kps = self.__apply_sift(img)
        
        # Convert the watermark image to binary
        adjusted_watermark = self.__adjust_watermark(watermark)

        # Iterate through all selected kps and embed watermark
        for kp in n_kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])

            # Transform watermark patch to match keypoint's scale and orientation
            transformed_wm = self.__apply_wm_transform(adjusted_watermark, kp)

            # Calculate patch size and offsets
            h_patch, w_patch = transformed_wm.shape
            
            x_diff = w_patch // 2
            y_diff = h_patch // 2

            # Extract blue channel patch centered at keypoint
            submatrix = img[
                (y - y_diff):(y + y_diff + 1), 
                (x - x_diff):(x + x_diff + 1), 
                0
            ]
            
            # Clear LSB of each pixel and embed watermark
            submatrix &= 254
            submatrix |= transformed_wm.astype(np.uint8)

        return img

    def recover(self, img, watermark):
        """
        Attempts to recover a previously embedded binary watermark from an image.
        Recovery succeeds only if all patches match the expected watermark exactly.
        
        Returns:
            - The success status of the recovery process
        """

        # Detect SIFT keypoints in provided image
        kps = self.__apply_sift(img)
        
        # Convert the watermark image to binary
        adjusted_watermark = self.__adjust_watermark(watermark)

        # Compare expected and extracted patches at each keypoint
        for kp in kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
        
            # Extract the LSB patch around the keypoint
            patch = self.__get_kp_patch(img, kp)

            # Apply the same transform used during embedding
            transformed_wm = self.__apply_wm_transform(adjusted_watermark, kp)

            # If any patch doesn't match exactly recovery fails
            if not np.array_equal(patch, transformed_wm):
                return False

        return True

    def __get_kp_patch(self, img, kp):
        """
        Extracts the LSB patch from the blue channel given a keypoint.
        The patch size is dynamically determined based on the keypoint's scale.
        """

        # Determine patch size based on the keypoint's scale
        scale, patch_size = self.__scale_kp(kp)

        # Get the LSB of the blue channel
        img_lsb = img[..., 0] & 1

        x, y = int(kp.pt[0]), int(kp.pt[1])
        half_size = patch_size // 2

        # Extract a patch centered on the keypoint
        patch = img_lsb[
            max(y - half_size, 0):y + half_size + 1,
            max(x - half_size, 0):x + half_size + 1
        ]
        return patch

    def tampered(self, img, watermark):
        """
        Checks whether an image has been tampered with by comparing extracted watermark patches
        to the expected watermark.

        Returns:
            - Proportion of keypoints fully verified
            - Inverse of average Hamming similarity
            - Image with overlaid keypoint verification status
        """

        # Separate BGR and alpha channels
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        alpha = img[..., 3]

        # Create a working copy for visualization
        img_copy = bgr.copy()

        # Detect SIFT keypoints in the image
        kps = self.__apply_sift(img)

        # Convert the watermark image to binary
        adjusted_watermark = self.__adjust_watermark(watermark)

        all_similarities = []
        verified_count = 0

        # Compare expected and extracted patches at each keypoint
        for kp in kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])

            # Extract patch from image and generate expected transformed watermark
            patch = self.__get_kp_patch(img, kp)
            transformed_wm = self.__apply_wm_transform(adjusted_watermark, kp)

            # Calculate Hamming similarity between patch and expected watermark
            similarity = self.__calc_patch_similarity(patch, transformed_wm)
            all_similarities.append(similarity)

            # Choose marker color based on similarity level
            if similarity == 0:
                verified_count += 1
                color = (0, 255, 0)  # Green
            elif similarity <= 0.4:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red

            # Draw circle on output image, coloured based on similarity
            cv2.circle(img_copy, (x, y), radius=4, color=color, thickness=2)

        # Reattach alpha channel
        img_output = cv2.merge((img_copy, alpha))

        # Calculate similarity metrics
        if all_similarities:
            avg_similarity = round(np.mean(all_similarities), 3)
            verified = round((verified_count / len(kps)), 2)
        else:
            avg_similarity = 1.0
            verified = 1

        return verified, (1 - avg_similarity), img_output

    def __apply_wm_transform(self, img, kp):
        """
        Applies a scale and rotation transformation to the binary watermark to
        align with the scale and orientation of the given keypoint.
        """

        # Compute scale and patch size from keypoint
        scale, patch_size = self.__scale_kp(kp)

        # Resize watermark to match patch size
        img = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
        center = (patch_size // 2, patch_size // 2)
        
        # Round keypoint angle to the nearest 45 degrees for better consistency
        angle = round(kp.angle / 45) * 45

        # Compute affine rotation matrix using keypoint's angle and scale
        M = cv2.getRotationMatrix2D(center, angle, scale)

        # Adjust the transformation matrix to keep the output centered
        output_center = (patch_size // 2, patch_size // 2)
        M[0, 2] += output_center[0] - center[0]
        M[1, 2] += output_center[1] - center[1]

        # Apply affine transformation to the patch
        transformed_patch = cv2.warpAffine(img, M, (patch_size, patch_size), flags=cv2.INTER_NEAREST)
        
        # Convert result to binary
        transformed_patch = (transformed_patch * 255).astype(np.uint8)
        _, transformed_patch = cv2.threshold(transformed_patch, 127, 1, cv2.THRESH_BINARY)

        return transformed_patch

    def __apply_sift(self, img):
        """
        Detects SIFT keypoints in an image.
        Ensures that selected keypoints are not too close together and that patches remain
        within image bounds.
        """

        # Convert image to grayscale
        img_gray = self.__img_to_grayscale(img)

        # Detect keypoints using SIFT
        sift = cv2.SIFT_create()
        kps = sift.detect(img_gray, None)

        h, w = img_gray.shape[:2]    

        # Sort by strongest response
        kps = sorted(kps, key=lambda kp: -kp.response)

        valid_kps = []
        for kp in kps:
            # Determine patch size based on keypoint scale
            scale, patch_size = self.__scale_kp(kp)
            half_size = patch_size // 2

            x, y = int(kp.pt[0]), int(kp.pt[1])

            # Skip keypoints too close to image boundaries
            if (x - half_size < 0 or x + half_size >= w or 
                y - half_size < 0 or y + half_size >= h):
                continue

            too_close = False
            for v_kp in valid_kps:
                # Measure distance between keypoints based on patch size
                dx = kp.pt[0] - v_kp.pt[0]
                dy = kp.pt[1] - v_kp.pt[1]

                # Ignore if keypoint is too close to an existing one
                if dx * dx + dy * dy < (half_size * 2) ** 2:
                    too_close = True
                    break

            # Add keypoint if separated
            if not too_close:
                valid_kps.append(kp)

            # Stop if desired number of keypoints is reached
            if len(valid_kps) == self.N_KEYPOINTS:
                break

        return valid_kps

    def __scale_kp(self, kp):
        """
        Calculates the appropriate scale and patch size for a given keypoint.
        """
    
        # Compute scale factor relative to the base patch size
        scale = kp.size / self.BASE_PATCH_SIZE
        
        # Scale the patch size and round to nearest multiple of step
        step = 2
        patch_size = int(np.ceil(self.BASE_PATCH_SIZE * scale))
        patch_size = round(patch_size / step) * step
            
        # Ensure patch size is odd to ensure symmetry
        if patch_size % 2 == 0:
            patch_size += 1

        return scale, patch_size

    def __adjust_watermark(self, img):
        """
        Converts the input watermark image to a binary matrix
        """
        img_gray = self.__img_to_grayscale(img)
        _, img_binary = cv2.threshold(img_gray, 127, 1, cv2.THRESH_BINARY)
        return img_binary

    def __calc_patch_similarity(self, patch, wm):
        """
        Calculates the Hamming similarity between a recovered patch and the expected watermark patch.
        """
        hamming = np.sum(patch != wm)
        score = hamming / wm.size

        return score

    def __img_to_grayscale(self, img):
        """
        Converts an input image to grayscale, handling different channel configurations
        """
        if len(img.shape) == 2:
            return img
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
