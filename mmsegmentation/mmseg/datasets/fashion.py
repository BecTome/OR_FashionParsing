from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

palette = [
    (0, 0, 0), (67, 161, 255), (167, 146, 11), (136, 126, 185), (44, 52, 10), (25, 33, 189), (73, 197, 184),
    (20, 165, 16), (48, 37, 106), (98, 213, 120), (21, 104, 190), (191, 106, 197), (142, 63, 109), (155, 22, 122),
    (43, 152, 125), (128, 89, 85), (11, 1, 133), (126, 45, 174), (32, 111, 29), (55, 31, 198), (70, 250, 116),
    (216, 21, 138), (100, 0, 176), (171, 236, 47), (193, 137, 224), (36, 152, 214), (154, 165, 67), (73, 8, 110),
    (67, 161, 255), (167, 146, 11), (136, 126, 185), (44, 52, 10), (25, 33, 189), (73, 197, 184), (20, 165, 16),
    (48, 37, 106), (98, 213, 120), (21, 104, 190), (191, 106, 197), (142, 63, 109), (155, 22, 122), (43, 152, 125),
    (128, 89, 85), (11, 1, 133), (126, 45, 174), (32, 111, 29), (55, 31, 198)#, (70, 250, 116), (216, 21, 138)
]

# This is the same as above but with the background
d_cats_bg = {0: 'background', 1: 'shirt, blouse', 2: 'top, t-shirt, sweatshirt', 3: 'sweater', 4: 'cardigan', 5: 'jacket', 6: 'vest', 7: 'pants', 
             8: 'shorts', 9: 'skirt', 10: 'coat', 11: 'dress', 12: 'jumpsuit', 13: 'cape', 14: 'glasses', 15: 'hat', 16: 'headband, head covering, hair accessory', 
             17: 'tie', 18: 'glove', 19: 'watch', 20: 'belt', 21: 'leg warmer', 22: 'tights, stockings', 23: 'sock', 24: 'shoe', 25: 'bag, wallet', 26: 'scarf', 
             27: 'umbrella', 28: 'hood', 29: 'collar', 30: 'lapel', 31: 'epaulette', 32: 'sleeve', 33: 'pocket', 34: 'neckline', 35: 'buckle', 36: 'zipper', 
             37: 'applique', 38: 'bead', 39: 'bow', 40: 'flower', 41: 'fringe', 42: 'ribbon', 43: 'rivet', 44: 'ruffle', 45: 'sequin', 46: 'tassel'}

classes = list(d_cats_bg.values())

@DATASETS.register_module()
class FashionBG(BaseSegDataset):
  METAINFO = dict(classes = classes, palette = palette)
  def __init__(self, **kwargs):
    # remove datasets from kwargs if it exists
    #if 'datasets' in kwargs:
    #  kwargs.pop('datasets')
    super().__init__(img_suffix='.jpg', seg_map_suffix='_seg.png', **kwargs)