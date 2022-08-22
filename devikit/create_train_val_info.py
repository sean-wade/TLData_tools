"""
    功能: 
        针对多文件夹组织的 SGData
        为每个文件夹生成专属的 info.json
            (如 train_001_20220806_infos.json 和 val_001_20220806_infos.json)
        最后合并成总的 info.pkl
            (如 train_total_infos.json 和 val_total_infos.json)

    使用:
        python create_train_val_info.py \
            --sg_data_dir /mnt/data/SGTrain \
            --save_path /mnt/data/SGTrain/tl_infos \
            --ratio 0.85 \
            --exclude 001_20220801 \
            --exclude 002_20220809
    
    参数说明: 
        --sg_data_dir, SG5 的数据集路径, 其下面是类似于 001_20220806 的文件夹, 该文件夹中至少包含(f30_undist & labelled_trafficlights) 两个文件夹
        --save_path, infos 和 日志 保存的目录, 由于训练平台的数据盘没有写入权限, 请保存至其他位置
        --exclude, 生成时排除的子文件夹, (如 001_20220801 文件夹中的 json 文件标注格式不对，且与 001_20220806 重复, 因此需要排除)
"""
import os
import tqdm
import json
import glob
import torch
import shutil
import random
import logging
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt


# 确保每次随机 sample 的序列相同
random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description="ceate sg data info")
    parser.add_argument("--sg_data_dir", type=str, default="/home/jovyan/vol-2", help="path to sg data directory")
    parser.add_argument("--save_path", type=str, default="/home/jovyan/workspace/infos", help="path to the infos saving directory" )
    parser.add_argument("--ratio", type=float, default=0.8, help="split ratio of train/all")
    parser.add_argument("--exclude", action='append', default=["001_20220801"], help="sub name to exclude")
    args = parser.parse_args()
    return args


def check_paths_time_consistency(data_paths, anno_paths):
    if len(data_paths) != len(anno_paths):
        logging.error("Error, paths length aren't the same %d, %d ! " % (len(data_paths), len(anno_paths) ) )
        return False
    for dp,ap in zip(data_paths, anno_paths):
        name1 =  os.path.basename(dp)
        name2 = os.path.basename(ap)
        if ".".join(name1.split('.')[:2]) != ".".join(name2.split('.')[:2]):
                logging.error("Error, different timestamp between %s and %s" %(name1, name2))
                return False
    return True


def get_valid_subdirs(dir, exclude):
    """
        获取目录下存在 f30_undist 和 labelled_trafficlights 的文件夹列表
    """
    valid_subdirs = []
    all_subfolders = os.listdir(dir)

    for subfolder in all_subfolders:
        if subfolder in exclude:
            continue
        
        subpath = dir + "/" + subfolder
        if os.path.isdir(subpath):
            tmp_fs = os.listdir(subpath)
            if "labelled_trafficlights" in tmp_fs and "f30_undist" in tmp_fs:
                valid_subdirs.append(subpath)
    return valid_subdirs


def get_all_package_ids(label_dir):
    """
        获取某个 label_dir 下的所有 package_ids
    """
    package_ids = []
    labels = os.listdir(label_dir)
    for label_name in labels:
        package_ids.append(json.load(open(label_dir+"/"+label_name,'r'))["infos"]["tl_package_id"])
    return list(set(package_ids))


def random_split_ids(id_list, ratio=0.8):
    id_nums = len(id_list)
    id_nums_train = int(ratio * id_nums)
    logging.info(f"    Total package nums = {id_nums}, train nums = {id_nums_train}")
    train_ids = random.sample(id_list, id_nums_train)
    val_ids = [iiid for iiid in id_list if iiid not in train_ids]
    return train_ids, val_ids


def get_files_by_pkgids(pkg_ids, subdir):
    label_dir = os.path.join(subdir, "labelled_trafficlights")
    jpg_dir = os.path.join(subdir, "f30_undist")
    label_paths, jpg_paths = [], []
    label_names = os.listdir(label_dir)
    for label_name in label_names:
        label_full_path = os.path.join(label_dir, label_name)
        jpg_full_path = os.path.join(jpg_dir, label_name.replace(".json", ".jpg"))
        cur_pkg_id = json.load(open(label_dir + "/" + label_name,'r'))["infos"]["tl_package_id"]
        if cur_pkg_id in pkg_ids:
            label_paths.append(label_full_path)
            jpg_paths.append(jpg_full_path)
    return label_paths, jpg_paths


def main(args):
    valid_subdirs = get_valid_subdirs(args.sg_data_dir, args.exclude)
    logging.info("Found valid subdirs : %s"%str(valid_subdirs))

    total_train_label_ps, total_train_jpg_ps = [], []
    total_val_label_ps, total_val_jpg_ps = [], []
    total_train_ids, total_val_ids = [], []

    logging.info("Generating train and val infos by create_train_val_info.py.")
    logging.info("SG data dir : %s\n"%args.sg_data_dir)

    for subdir in valid_subdirs:
        curr_package_ids = get_all_package_ids(subdir + "/labelled_trafficlights")
        logging.info(f"Process {subdir}")

        train_ids, val_ids = random_split_ids(curr_package_ids, ratio=args.ratio)
        logging.info("    Train package ids : %s" % str(train_ids))
        logging.info("    Val package ids : %s" % str(val_ids))

        train_label_paths, train_jpg_paths = get_files_by_pkgids(train_ids, subdir)
        val_label_paths, val_jpg_paths = get_files_by_pkgids(val_ids, subdir)

        sub_name = subdir.split("/")[-1]

        train_data = {
            "sg_data_dir"       : args.sg_data_dir,
            "nums"              : len(train_label_paths), 
            "tl_package_ids"    : train_ids,
            "images"            : train_jpg_paths,
            "annos"             : train_label_paths,
        }
        val_data = {
            "sg_data_dir"       : args.sg_data_dir,
            "nums"              : len(val_label_paths), 
            "tl_package_ids"    : val_ids,
            "images"            : val_jpg_paths,
            "annos"             : val_label_paths,
        }
        with open(args.save_path + f"/{sub_name}_train_infos.json", "w") as dump_f:
            json.dump(train_data, dump_f, indent=4)
        with open(args.save_path + f"/{sub_name}_val_infos.json", "w") as dump_f:
            json.dump(val_data, dump_f, indent=4)

        logging.info("Saved " + args.save_path + f"/{sub_name}_train_infos.json")
        logging.info("Saved " + args.save_path + f"/{sub_name}_val_infos.json")

        total_train_label_ps.extend(train_label_paths)
        total_train_jpg_ps.extend(train_jpg_paths)
        total_val_label_ps.extend(val_label_paths)
        total_val_jpg_ps.extend(val_jpg_paths)

        train_ids = [subdir + ":%s"%iii for iii in train_ids]
        val_ids = [subdir + ":%s"%iii for iii in val_ids]
        total_train_ids.extend(train_ids)
        total_val_ids.extend(val_ids)

        logging.info("Finished ... \n\n\n")

    logging.info("Start to process total infos.")

    all_train_data = {
        "sg_data_dir"       : args.sg_data_dir,
        "nums"              : len(total_train_label_ps), 
        "tl_package_ids"    : total_train_ids,
        "images"            : total_train_jpg_ps,
        "annos"             : total_train_label_ps,
    }
    all_val_data = {
        "sg_data_dir"       : args.sg_data_dir,
        "nums"              : len(total_val_label_ps), 
        "tl_package_ids"    : total_val_ids,
        "images"            : total_val_jpg_ps,
        "annos"             : total_val_label_ps,
    }
    with open(args.save_path + f"/total_train_infos.json", "w") as dump_f:
        json.dump(all_train_data, dump_f, indent=4)
    with open(args.save_path + f"/total_val_infos.json", "w") as dump_f:
        json.dump(all_val_data, dump_f, indent=4)

    logging.info("Saved total_train_infos.json and total_val_infos.")


def set_logger():
    logging.basicConfig(level    = logging.INFO,                                                     
                        format   = '%(asctime)s  %(filename)s : %(levelname)s  %(message)s', 
                        datefmt  = '%Y-%m-%d %H:%M:%S',
                        filename = args.save_path + "/generate_infos.log",
                        filemode = 'w') 
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  %(filename)s : %(levelname)s  %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    set_logger()
    main(args)
