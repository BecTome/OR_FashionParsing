import random
import numpy as np
import cv2
import os
import mmcv
from mmcv.transforms import BaseTransform, TRANSFORMS

@TRANSFORMS.register_module()# 
class AddPascalCrop(BaseTransform):
    def __init__(self, pascal_dataset_path = 'datasets/VOCdevkit/VOC2012', prob=0.5):
        super().__init__()
        self.dataset_path = pascal_dataset_path
        self.prob = prob
        
        self.images_dir = os.path.join(self.dataset_path, 'JPEGImages')
        self.segmentation_dir = os.path.join(self.dataset_path, 'SegmentationClass')
        

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results
        
        img = results['img']
        mask = results['mask']
        
        img_pascal, mask_pascal = self.get_random_image_pascal()
        
        width, height = img.shape[:2]
        width_pascal, height_pascal = img_pascal.shape[:2]
        
        
        cropped_image_pascal = img_pascal.copy()
        mask_pascal = mask_pascal.astype(bool)
        cropped_image_pascal[~mask_pascal] = 0
        
        # Crop the cropped img so that it has no black borders (rows and columns)
        min_x = np.min(np.where(mask_pascal)[1])
        max_x = np.max(np.where(mask_pascal)[1])
        min_y = np.min(np.where(mask_pascal)[0])
        max_y = np.max(np.where(mask_pascal)[0])
        
        cropped_image_pascal = cropped_image_pascal[min_y:max_y, min_x:max_x]
        mask_pascal = mask_pascal[min_y:max_y, min_x:max_x]
        width_pascal, height_pascal = cropped_image_pascal.shape[:2]
        

         ### ROTATION ###
        # Add some random rotation to the Pascal image
        angle = np.random.randint(-45, 45)
        cropped_image_pascal = self.rotate_image_without_cropping(cropped_image_pascal, angle)
        mask_pascal = self.rotate_image_without_cropping(mask_pascal.astype(np.uint8), angle)
        mask_pascal = mask_pascal.astype(bool)
        
        width_pascal, height_pascal = cropped_image_pascal.shape[:2]
        ##############################################

        
        ### RESIZE ###
        # Resize randomly the Pascal img
        min_scale = 0.8
        max_scale = min(width / width_pascal, height / height_pascal)
        
        scale = np.random.uniform(min_scale, max_scale)
        width_pascal = int(width_pascal * scale)
        height_pascal = int(height_pascal * scale)
        
        
        cropped_image_pascal = cv2.resize(cropped_image_pascal, (height_pascal, width_pascal))
        mask_pascal = cv2.resize(mask_pascal.astype(np.uint8), (height_pascal, width_pascal))
        mask_pascal = mask_pascal.astype(bool)
        width_pascal, height_pascal = cropped_image_pascal.shape[:2]
        ##############################################
        
        ### OFFSET ###
        # Calculate center position for superposition
        x = (width - width_pascal) // 2
        y = (height - height_pascal) // 2
        
        # Add some random offset to the center position
        offset_x = np.random.randint(-x//3, x//3) if x > 0 else 0
        offset_y = np.random.randint(-y//2, y//2) if y > 0 else 0
        
        x, y = x + offset_x, y + offset_y
        ##############################################
        
        
        # Superpose Pascal img on the original img at the calculated center position
        overlaid_image = img.copy()
        
        # Create a mask for the Pascal img where mask_pascal is True
        pascal_mask = np.zeros_like(mask, dtype=bool)
        pascal_mask[x:x+width_pascal, y:y+height_pascal] = mask_pascal
        
        # Overlay Pascal img on the original img at the calculated center position
        overlaid_image = img.copy()
        overlaid_image[pascal_mask] = cropped_image_pascal[mask_pascal]
        
        overlaid_mask = mask.copy() 
        overlaid_mask[pascal_mask] = 0   
        
        
        results['img'] = overlaid_image
        results['mask'] = overlaid_mask        
        return results
    
    def get_random_image_pascal(self):
        # Get a random img from the dataset and return the img and the mask
        mask_files = os.listdir(self.segmentation_dir)
        rnd_file = np.random.choice(mask_files, 1)[0]
        mask_file = os.path.join(self.segmentation_dir, rnd_file)
        image_file = os.path.join(self.images_dir, rnd_file[:-4] + '.jpg')
        
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print("img file:", image_file)
        # print("Mask file:", mask_file)
        mask = cv2.imread(mask_file)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        mask = self.make_boolean_mask(mask)

        
        return img, mask
    
    @staticmethod
    def make_boolean_mask(mask):
        avoid_colors = np.array([[0, 0, 0], [224, 224, 192]])
        
        # Get unique colors in the mask
        colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
        
        # Create a boolean mask to identify rows not in avoid_colors
        mask_colors = ~np.isin(colors, avoid_colors).all(axis=1)
        filtered_colors = colors[mask_colors]

        # Get one random colors from filtered colors
        if filtered_colors.shape[0] == 0:
            color = colors[np.random.choice(colors.shape[0], 1)[0]]
        else:
            color = filtered_colors[np.random.choice(filtered_colors.shape[0], 1)[0]]
        
        mask = np.all(mask == color, axis=2)    
        
        return mask
    
    @staticmethod
    def rotate_image_without_cropping(img, angle):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        # Calculate the size of the new image
        abs_cos, abs_sin = abs(np.cos(np.radians(angle))), abs(np.sin(np.radians(angle)))
        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)
        # Adjust the rotation matrix to the center and apply the padding
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotation_matrix[0, 2] += bound_w / 2 - center[0]
        rotation_matrix[1, 2] += bound_h / 2 - center[1]
        rotated_img = cv2.warpAffine(img, rotation_matrix, (bound_w, bound_h))
        return rotated_img