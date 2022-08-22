import os
import json


# revise "unknow" -> "unknown"
def revise(in_json, out_json):
    tt = json.load(open(in_json))
    for obj in tt["objects"]:
        for k,v in obj.items():
            if v == "unknow":
                obj[k] = "unknown"
            
            if k in ["child_color", "child_shape"]:
                for i,vv in enumerate(v):
                    if vv == "unknow":
                        obj[k][i] = "unknown"


    with open(out_json, "w") as dump_f:
        json.dump(tt, dump_f, indent=4)


dir_src = "/mnt/data/SGData/TLS_dataset/traffic_light_2022_07_19_13_28/camera_f30/annotations_ori"
dir_dst = "/mnt/data/SGData/TLS_dataset/traffic_light_2022_07_19_13_28/camera_f30/annotations"

fs = os.listdir(dir_src)

for ff in fs:
    revise(dir_src+"/"+ff, dir_dst+"/"+ff)



