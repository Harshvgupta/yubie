# yolo task=detect \
#     mode=predict \
#     model='/Users/pranomvignesh/Workfolder/yubie/fine-tuned/best.pt' \
#     conf=0.25 \
#     source='/Users/pranomvignesh/Workfolder/yubie/fetch/ball/cimages/left/output_video_left.mp4' \
#     save=True

yolo task=detect \
    mode=predict \
    model='/Users/pranomvignesh/Workfolder/yubie/fine-tuned/best.pt' \
    conf=0.25 \
    source='/Users/pranomvignesh/Workfolder/yubie/fetch/ball/cimages/right/output_video_right.mp4' \
    save=True