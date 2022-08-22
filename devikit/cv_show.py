import os
import cv2
import numpy as np

from yolox.data.datasets import SGTLS_Detection, SGTLS_ATTRIBUTES


def vis_npy_targets(img, targets):
    img1 = img.astype(np.uint8).copy()

    for ttt in targets:
        ttt = ttt.astype(np.int)
        # lt = (int(ttt[1] - ttt[3]/2.0), int(ttt[2] - ttt[4]/2.0))
        # rb = (int(ttt[1] + ttt[3]/2.0), int(ttt[2] + ttt[4]/2.0))
        lt = (ttt[0], ttt[1])
        rb = (ttt[2], ttt[3])
        img1 = cv2.rectangle(img1, lt, rb, (0,255,0),2)

        if len(ttt) > 5:
            attr_str = "".join(ttt[5:].astype(np.str))
            cv2.putText(img1, attr_str, lt, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1)
    
    return img1


colors = [tuple(map(int, np.random.randint(120, high=250, size=(3,)))) for _ in range(100)]
def vis_str_targets(img0, output, bboxes, attr_names_list):
    img = img0.astype(np.uint8).copy()
    for idx, (out, bbox) in enumerate(zip(output, bboxes)):
        x0 = int(bbox[0])
        y0 = int(bbox[1])
        x1 = int(bbox[2])
        y1 = int(bbox[3])

        out = out.astype(np.int)

        color = colors[idx]
        # color = colors[idx]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
        cv2.rectangle(img, (x0, y1+1), (x0+80, y1+120), color, 2)

        for attr_idx, attr_class_id in enumerate(out):
            attr_str = attr_names_list[attr_idx][attr_class_id]
            cv2.putText(img, attr_str, (x0, y1+13*attr_idx+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def show_attr_num(data_root):
    sgtls = SGTLS_Detection(data_root)
    for i in range(len(sgtls)):
        img, target, img_info, idx = sgtls.pull_item(i)
        img_show = vis_npy_targets(img, target)
        cv2.imshow("r", img_show)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()


def show_attr_str(data_root):
    sgtls = SGTLS_Detection(data_root)
    attr_names_list = list(SGTLS_ATTRIBUTES.values())

    for i in range(len(sgtls)):
        img, target, img_info, idx = sgtls.pull_item(i)
        img_show = vis_str_targets(img, target[:, 5:], target[:, 0:4], attr_names_list)
        cv2.imshow("r", img_show)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # data_root = "/mnt/data/SGData/TLS_dataset/traffic_light_2022_07_19_13_28/camera_f30/"
    data_root = "/mnt/data/SGTrain/TLS_dataset/traffic_light_all/"
    # show_attr_num(data_root)
    show_attr_str(data_root)
    
