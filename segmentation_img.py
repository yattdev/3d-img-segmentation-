#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pixellib
from pixellib.instance import instance_segmentation

segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5") 
target_classes = segment_image.select_target_classes(BG = True, person=True, bicycle=True, car=True, motorcycle=True, airplane=True,
                      bus=True, train=True, truck=True, boat=True, traffic_light=True, fire_hydrant=True,
                      stop_sign=True,
                      parking_meter=True, bench=True, bird=True, cat=True, dog=True, horse=True, sheep=True,
                      cow=True, elephant=True, bear=True, zebra=True,
                      giraffe=True, backpack=True, umbrella=True, handbag=True, tie=True, suitcase=True,
                      frisbee=True, skis=True, snowboard=True,
                      sports_ball=True, kite=True, baseball_bat=True, baseball_glove=True, skateboard=True,
                      surfboard=True, tennis_racket=True,
                      bottle=True, wine_glass=True, cup=True, fork=True, knife=True, spoon=True, bowl=True,
                      banana=True, apple=True, sandwich=True, orange=True,
                      broccoli=True, carrot=True, hot_dog=True, pizza=True, donut=True, cake=True, chair=True,
                      couch=True, potted_plant=True, bed=True,
                      dining_table=True, toilet=True, tv=True, laptop=True, mouse=True, remote=True,
                      keyboard=True, cell_phone=True, microwave=True,
                      oven=True, toaster=True, sink=True, refrigerator=True, book=True, clock=True, vase=True,
                      scissors=True, teddy_bear=True, hair_dryer=True,
                      toothbrush=True)
segment_image.segmentImage("test.jpeg", extract_segmented_objects=True, output_image_name="output.jpeg", show_bboxes=True, segment_target_classes=target_classes)
