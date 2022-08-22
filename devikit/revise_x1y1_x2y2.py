import os
import json


# revise "unknow" -> "unknown"
def revise(in_json, out_json):
    tt = json.load(open(in_json))
    for obj in tt["objects"]:
        box = obj["bbox"]
        new_box = [box[2], box[3], box[0], box[1]]
        obj["bbox"] = new_box

    with open(out_json, "w") as dump_f:
        json.dump(tt, dump_f, indent=4)


dir_src = "/mnt/data/SGTrain/TLS_dataset/traffic_light_2022_07_19_13_28/camera_f30/labelled_trafficlights"
dir_dst = "/mnt/data/SGTrain/TLS_dataset/traffic_light_2022_07_19_13_28/camera_f30/labelled_trafficlights_xyxy"

fs = os.listdir(dir_src)

for ff in fs:
    revise(dir_src+"/"+ff, dir_dst+"/"+ff)



