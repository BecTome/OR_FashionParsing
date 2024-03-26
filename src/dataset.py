from pycocotools.coco import COCO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from src.config import COLORS, d_cats, dataDir, dataType, maskFolder, imgFolder, annFile, db

def get_image(img_id):
    img = db.loadImgs(img_id)[0]
    I = io.imread('{}/images/{}/{}'.format(dataDir,dataType,img['file_name']))
    return I

def get_mask(img_id):
    file_name = db.loadImgs(img_id)[0]['file_name']
    mask = cv2.imread(f"{maskFolder}/{file_name.split('.')[0]}_seg.png", cv2.IMREAD_GRAYSCALE)
    return mask

def get_annotations(img_id):
    annIds = db.getAnnIds(imgIds=img_id)
    anns = db.loadAnns(annIds)
    return anns

def box_label(image, box, mask=None, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  if mask is not None:
    image = image * (1-mask) + (image + np.array([[list(color)]])) * mask / 2
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  image = cv2.rectangle(image.astype(np.uint8), p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    image = cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    image = cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)
    return image
  
def plot_boxes(img_id, ax=None):
    img = get_image(img_id)
    anns = get_annotations(img_id)

    plt.imshow(img)
    for ann in anns:
        color = COLORS[ann['category_id']]
        bbox = ann['bbox']
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        img = box_label(img, bbox, label=db.loadCats(ann['category_id'])[0]['name'], color=color)

    if ax is not None:
        ax.imshow(img)
        ax.axis('off')
    else:
        plt.imshow(img)
        plt.axis('off')

def plot_mask(mask_id):
    mask = cv2.imread(f"{maskFolder}/{mask_id}", cv2.IMREAD_GRAYSCALE)
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    # plt.show()

def overlay_mask(img_file, alpha=0.7, ax=None):
    mask = cv2.imread(f"{maskFolder}/{img_file.split('.')[0]}_seg.png", cv2.IMREAD_GRAYSCALE)
    img = plt.imread(f"{imgFolder}/{img_file}", cv2.IMREAD_COLOR)

    # For all the values in the mask, we will create a new mask with the same shape but with the color of the category
    mask_color = np.zeros((mask.shape[0], mask.shape[1], 3))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask_color[i, j] = COLORS[mask[i, j]]

    mask_color = mask_color.astype(np.uint8)

    # Combine original image with colored mask
    image = cv2.addWeighted(img, 1 - alpha, mask_color, alpha, 0)

    # image = cv2.addWeighted(img, 1, mask_color, 1, 0)

    if ax is not None:
        ax.imshow(image)
        ax.axis('off')
    else:
        # plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
  