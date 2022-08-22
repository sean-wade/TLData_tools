import os
import json
import numpy as np


num_int = {
    "one" : 1, 
    "two" : 2,
    "three" : 3,
    "four" : 4,
    "unknown" : -1
}

tl_colors = ["red", "green", "yellow", "dark", "unknown"]
picts = [
            "circle", "arrow_straight", "arrow_left", "arrow_right", "arrow_uturn", 
            "arrow_straight_left", "arrow_straight_right", "arrow_uturn_left", 
            "bicycle", "pedestrian", 
            "lane_stop", "lane_straight", 
            "digit", 
            "unknown"
    ]


def get_obj_color(obj):
    if obj["state"] == "unknown":
        return "unknown"

    elif obj["state"] == "off":
        return "dark"
    
    else:
        child_num = num_int[obj["child_num"]]
        if child_num < 0:
            print("    **** Error, child_num unknown !")
            return "unknown"
        else:
            color_nums = np.array([obj["child_color"].count(cc) for cc in tl_colors])
            if color_nums[:3].sum() > 0:
                # r/g/y
                return tl_colors[np.argmax(color_nums[:3])]
            elif color_nums[3] > 0:
                return "dark"
            else:
                return "unknown"


def get_obj_pict(obj):
    if obj["state"] in ["unknown", "off"]:
        return "unknown"

    child_num = num_int[obj["child_num"]]
    if child_num < 0:
        print("    **** Error, child_num unknown !")
        return "unknown"
    
    new_child_shapes = obj["child_shape"].copy()
    for i,new_child_shape in enumerate(new_child_shapes):
        if new_child_shape.startswith("digit"):
            new_child_shapes[i] = "digit"

    pict_counts = np.array([new_child_shapes.count(cc) for cc in picts])
    if pict_counts[-1] == child_num:
        # all unknown
        return "unknown"

    pict_counts[-1] = -1
    max_idx = np.argmax(pict_counts)
    max_num = pict_counts[max_idx]
    if max_num == 0:
        return "unknown"

    # 可能出现: 如4个子灯中，第一个是左转，第四个是倒计时
    # 但是目前标注公司导出的文件有问题，第四个灯的信息没有导出
    pict_counts[max_idx] = -1
    second_maxidx = np.argmax(pict_counts)
    second_max_num = pict_counts[second_maxidx]
    if second_max_num > 0:
        print("    **** Warning, at least two pict !!!")

    return picts[max_idx]


def add_color_pict_node(in_json, out_json):
    tt = json.load(open(in_json))
    # print(in_json)
    for obj in tt["objects"]:
        color = get_obj_color(obj)
        # # print(obj["child_color"], obj["state"], obj["child_num"], "====>>>>>", color)
        obj.update({"color" : color})

        pict = get_obj_pict(obj)
        # print(obj["child_shape"], obj["state"], obj["child_num"], "====>>>>>", pict)
        obj.update({"pict" : pict})

    with open(out_json, "w") as dump_f:
        json.dump(tt, dump_f, indent=4)


# src_path = "/mnt/data/SGData/TLS_dataset/traffic_light_2022_07_19_13_28/camera_f30/annotations"
# dst_path = "/mnt/data/SGData/TLS_dataset/traffic_light_2022_07_19_13_28/camera_f30/annotations_add"
src_path = "/mnt/data/SGTrain/TLS_dataset/traffic_light_20220810/labelled_trafficlights"
dst_path = "/mnt/data/SGTrain/TLS_dataset/traffic_light_20220810/labelled_trafficlights_add"

jsons = os.listdir(src_path)

for js in jsons:
    js_file = os.path.join(src_path, js)
    out_file = os.path.join(dst_path, js)
    # print(js_file)
    add_color_pict_node(js_file, out_file)