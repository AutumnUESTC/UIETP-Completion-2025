import argparse
import base64
import json
import os
import os.path as osp

import imgviz
import PIL.Image

from labelme.logger import logger
from labelme import utils

def main(json_file, file):
   logger.warning(
       "This script is aimed to demonstrate how to convert the "
       "JSON file to a single image dataset."
   )
   logger.warning(
       "It won't handle multiple JSON files to generate a "
       "real-use dataset."
   )


   out_dir = 'E:/yby/lab/Dataset'
   if not osp.exists(out_dir):
       os.mkdir(out_dir)

   data = json.load(open(json_file))
   imageData = data.get("imageData")

   if not imageData:
       imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
       with open(imagePath, "rb") as f:
           imageData = f.read()
           imageData = base64.b64encode(imageData).decode("utf-8")
   img = utils.img_b64_to_arr(imageData)


   label_name_to_value = {'_background_': 0,
                          'icedTower':1,
                          'icedCamera':2,
                          'icedCameraSnow':3,
                          'icedCameraFog':4,
                          "coverdCamera":5}
   # "icedTower", "icedCamera", "icedCameraSnow", "icedCameraFog", "coverdCamera"

   lbl, _ = utils.shapes_to_label(
       img.shape, data["shapes"], label_name_to_value
   )

   label_names = [None] * (max(label_name_to_value.values()) + 1)
   for name, value in label_name_to_value.items():
       label_names[value] = name

   lbl_viz = imgviz.label2rgb(
       label=lbl, image=imgviz.asgray(img), label_names=label_names, loc="rb"
   )

   # PIL.Image.fromarray(img).save(osp.join(out_dir + "img", file.split('.')[0] + ".png"))
   utils.lblsave(osp.join(out_dir + "/SegmentationClassPNG", file.split('.')[0] + ".png"), lbl)
   PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir + "/SegmentationClassVisualization", file.split('.')[0] + ".png"))

   with open(osp.join(out_dir, "label_names.txt"), "w") as f:
       for lbl_name in label_names:
           f.write(lbl_name + "\n")

   logger.info("Saved to: {}".format(out_dir))

if __name__ == "__main__":
    json_path = 'E:/yby/lab/IceDataset/before/SegmentationClass'

    for root, dirs, files in os.walk(json_path):
        for file in files:
            if file == '.DS_Store':
                continue
            json_path = os.path.join(root, file)
            print(json_path, file)
            main(json_path, file)
