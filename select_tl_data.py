import os
import cv2
import torch
import shutil
from tqdm import tqdm

from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import SGTLS_ATTRIBUTES


def get_model(exp_file, ckpt_file):
    exp = get_exp(exp_file)
    model = exp.get_model()
    print("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.cuda()
    model.eval()

    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    print("loaded checkpoint done.")
    return model


def find_obj_seq(obj_nums):
    obj_nums = obj_nums + [0]   # add last stop
    streak = 0
    seqs = []
    for i, num in enumerate(obj_nums):
        if num > 0:
            streak += 1
        else:
            if streak > 2:
                seqs.append([jj for jj in range(i-streak, i)])
            streak = 0
    return seqs


def main(img_path, model, out_dir, inteval=3):
    img_names = os.listdir(img_path)
    img_names.sort()
    tl_nums = [0] * len(img_names)

    preproc = ValTransform(legacy=False)
    for i, img_name in tqdm(enumerate(img_names)):
        img_ppp = os.path.join(img_path, img_name)
        img = cv2.imread(img_ppp)
        img, _ = preproc(img, None, (576,1024))
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float().cuda()
        with torch.no_grad():
            outputs = model(img)
            outputs = postprocess(
                outputs, 1, 0.6,
                0.45, class_agnostic=True, attr_nums=[len(attr_values) for attr_values in SGTLS_ATTRIBUTES.values()]
            )

            if outputs[0] is not None:
                tl_nums[i] = int(outputs[0].shape[0])
    
    os.makedirs(out_dir)
    with_obj_seqs = find_obj_seq(tl_nums)
    for i, seq in enumerate(with_obj_seqs):
        for idd in seq[::inteval]:
            src_img = os.path.join(img_path, img_names[idd])
            dst_img = os.path.join(out_dir, img_names[idd])
            shutil.copy(src_img, dst_img)
            


if __name__ == "__main__":
    model = get_model("/home/zhanghao/code/master/2_DET2D/YOLOX/exps/sgtls/sgtls_s.py", 
                      "/home/zhanghao/code/master/2_DET2D/YOLOX/YOLOX_outputs/sgtls_s_attr11_aug/best_ckpt.pth")

    main("/mnt/data/SGData/tl_0722_sb/f30", model, "/mnt/data/SGData/tl_0722_sb/f30_select")




