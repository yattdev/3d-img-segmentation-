import pixellib
#  from pixellib.torchbackend.instance import instanceSegmentation
from pixellib.instance import instanceSegmentation

ins = instanceSegmentation()
ins.load_model("mask_rcnn_coco.h5")
ins.segmentImage("github.jpg", show_bboxes=True, output_image_name="output_github2.jpg")
