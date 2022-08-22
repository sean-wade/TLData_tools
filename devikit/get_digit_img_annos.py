import os
import cv2
import json
from tqdm import tqdm
import os.path as osp


class DigitProcesser:
    """
        Function: crop sg-tls dataset's digit image and save in folder named number.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        print("data path : ", data_path)

        self.img_path = osp.join(data_path, "f30_undist")
        self.anno_path = osp.join(data_path, "labelled_trafficlights")
        assert osp.exists(self.img_path), f"{self.img_path} annos doesnot exist !"
        assert osp.exists(self.anno_path), f"{self.anno_path} imgs doesnot exist !"

        self.img_names = os.listdir(self.img_path)
        self.anno_names = os.listdir(self.anno_path)
        self.img_names.sort()
        self.anno_names.sort()

        assert len(self.img_names) == len(self.anno_names), "anno nums is not equal to img nums !"

        self.digit_img_path = osp.join(self.data_path, "digit")
        os.makedirs(self.digit_img_path, exist_ok=False)

    
    def split_to_one_folder(self):
        digit_big_count = 0
        for (anno_name, img_name) in tqdm(zip(self.anno_names, self.img_names)):
            print(anno_name)
            targets = json.load(open(osp.join(self.anno_path, anno_name)))
            img = cv2.imread(osp.join(self.img_path, img_name))

            for obj in targets["objects"]:
                if obj["direction"] != "front":
                    continue

                if obj["indication"] == "digit":
                    x1, y1, x2, y2 = int(obj["bbox"][0]),int(obj["bbox"][1]),int(obj["bbox"][2]),int(obj["bbox"][3])
                    obj_img = img[y1:y2, x1:x2]

                    child_shape = obj["child_shape"][0]
                    if child_shape == "unknown":
                        continue

                    digit = child_shape.split(":")[1].strip()

                    # 第一位没数字，补个 N 再继续
                    if len(digit)==1:
                        digit = "N" + digit
                    digit_num = 2 #len(digit)
                    
                    if obj_img is not None and obj_img.any():
                        split_width = (x2 - x1) // digit_num
                        print("digit=", digit)
                        for d_idx, ddd in enumerate(digit):
                            digit_img = obj_img[:, split_width*d_idx:split_width*(d_idx+1)]
                            # save_path = osp.join(self.digit_img_path, "%d_%d_num-%s.jpg"%(digit_big_count, d_idx, ddd))
                            save_path = osp.join(self.digit_img_path, "%s_%d_num-%s.jpg"%(anno_name[:-5], d_idx, ddd))
                            cv2.imwrite(save_path, digit_img)
                        
                        digit_big_count += 1
                    else:
                        print("    **** Error anno : ", anno_name, x1, y1, x2, y2)

        
    def split_to_10_folders(self):
        for i in range(10):
            os.makedirs(self.digit_img_path + "/%d"%i, exist_ok=True)
        os.makedirs(self.digit_img_path + "/N", exist_ok=True)

        digit_big_count = 0
        for (anno_name, img_name) in tqdm(zip(self.anno_names, self.img_names)):
            print(anno_name)
            targets = json.load(open(osp.join(self.anno_path, anno_name)))
            img = cv2.imread(osp.join(self.img_path, img_name))

            for obj in targets["objects"]:
                if obj["direction"] != "front":
                    continue

                if obj["indication"] == "digit":
                    x1, y1, x2, y2 = int(obj["bbox"][0]),int(obj["bbox"][1]),int(obj["bbox"][2]),int(obj["bbox"][3])
                    obj_img = img[y1:y2, x1:x2]
                    
                    child_shape = obj["child_shape"]
                    if len(child_shape)==0 or child_shape[0] == "unknown":
                        continue

                    digit = child_shape[0].split(":")[1].strip()

                    # 第一位没数字，补个 N 再继续
                    if len(digit)==1:
                        digit = "N" + digit
                    digit_num = 2 #len(digit)
                    
                    if obj_img is not None and obj_img.any():
                        split_width = (x2 - x1) // digit_num
                        print("digit=", digit)
                        for d_idx, ddd in enumerate(digit):
                            digit_img = obj_img[:, split_width*d_idx:split_width*(d_idx+1)]
                            save_path = osp.join(self.digit_img_path, "%s/%s_%d.jpg"%(ddd, anno_name[:-5], d_idx))
                            cv2.imwrite(save_path, digit_img)
                        
                        digit_big_count += 1
                    else:
                        print("    **** Error anno : ", anno_name, x1, y1, x2, y2)


root = "/mnt/data/SGTrain/TLS_dataset/traffic_light_all/"
cc = DigitProcesser(root)
# cc.split_to_one_folder()
cc.split_to_10_folders()

