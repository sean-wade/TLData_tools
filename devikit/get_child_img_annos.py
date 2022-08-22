import os
import cv2
import json
from tqdm import tqdm
import os.path as osp


class ChildCrop:
    """
        Function: crop sg-tls dataset's child light image and save annos(color & shape infos) as json.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        print("data path : ", data_path)

        self.img_path = osp.join(data_path, "images")
        self.anno_path = osp.join(data_path, "annotations")
        assert osp.exists(self.img_path), f"{self.img_path} annos doesnot exist !"
        assert osp.exists(self.anno_path), f"{self.anno_path} imgs doesnot exist !"

        self.img_names = os.listdir(self.img_path)
        self.anno_names = os.listdir(self.anno_path)
        self.img_names.sort()
        self.anno_names.sort()

        assert len(self.img_names) == len(self.anno_names), "anno nums is not equal to img nums !"

        self.crop_img_path = osp.join(self.data_path, "crop/imgs")
        self.crop_anno_path = osp.join(self.data_path, "crop/annos")
        os.makedirs(self.crop_img_path, exist_ok=False)
        os.makedirs(self.crop_anno_path, exist_ok=False)

    
    def crop(self):
        digit_big_count = 0
        child_count = 0
        for (anno_name, img_name) in tqdm(zip(self.anno_names, self.img_names)):
            print(anno_name)
            targets = json.load(open(osp.join(self.anno_path, anno_name)))
            img = cv2.imread(osp.join(self.img_path, img_name))

            for obj in targets["objects"]:
                if obj["direction"]=="back":
                    continue

                if obj["indication"] == "digit":
                    x2, y2, x1, y1 = int(obj["bbox"][0]),int(obj["bbox"][1]),int(obj["bbox"][2]),int(obj["bbox"][3])
                    obj_img = img[y1:y2, x1:x2]
                    
                    if obj_img is not None and obj_img.any():
                        save_path = osp.join(self.crop_img_path, "digit_big_%d.jpg"%digit_big_count)
                        cv2.imwrite(save_path, obj_img)
                        
                        out_json = osp.join(self.crop_anno_path, "digit_big_%d.json"%digit_big_count)
                        out_target = {
                            "shape" : obj["child_shape"][0],
                            "color" : obj["child_color"][0],
                        }
                        with open(out_json, "w") as dump_f:
                            json.dump(out_target, dump_f, indent=4)

                        digit_big_count += 1
                    else:
                        print("    **** Error anno : ", anno_name, x2, y2, x1, y1)

                else:
                    child_nums = min(len(obj["child_division"]), len(obj["child_color"]))    # 标注公司导出的 bug
                    if child_nums > 0:
                        for child_idx in range(child_nums):
                            x1 = int(obj["child_division"][child_idx][0])
                            y1 = int(obj["child_division"][child_idx][1])
                            x2 = int(obj["child_division"][child_idx][2])
                            y2 = int(obj["child_division"][child_idx][3])

                            obj_img = img[y1:y2, x1:x2]
                            if obj_img is not None and obj_img.any():
                                save_path = osp.join(self.crop_img_path, "child_%d.jpg"%child_count)
                                cv2.imwrite(save_path, obj_img)
                        
                                out_json = osp.join(self.crop_anno_path, "child_%d.json"%child_count)
                                out_target = {
                                    "shape" : obj["child_shape"][child_idx],
                                    "color" : obj["child_color"][child_idx],
                                }
                                with open(out_json, "w") as dump_f:
                                    json.dump(out_target, dump_f, indent=4)

                                child_count += 1
                            else:
                                print("    **** Error anno : ", anno_name, x1, y1, x2, y2)

root = "/mnt/data/SGData/TLS_dataset/traffic_light_2022_07_19_13_28/camera_f30/"
cc = ChildCrop(root)
cc.crop()

