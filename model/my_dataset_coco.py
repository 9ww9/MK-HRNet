import os
import copy

import torch
import numpy as np
import cv2
import torch.utils.data as data
from pycocotools.coco import COCO


class CocoKeypoint(data.Dataset):
    def __init__(self,
                 root,
                 dataset="train",
                 years="2025",
                 transforms=None,
                 det_json_path=None,
                 fixed_size=(256, 192)):
        super().__init__()
        assert dataset in ["train", "val"], 'dataset must be in ["train", "val"]'
        anno_file = f"ant_keypoints_{dataset}{years}.json"
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        self.img_root = os.path.join(root, f"{dataset}{years}")
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        self.anno_path = os.path.join(root, "annotations", anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.fixed_size = fixed_size
        self.mode = dataset
        self.transforms = transforms
        self.coco = COCO(self.anno_path)
        img_ids = list(sorted(self.coco.imgs.keys()))

        if det_json_path is not None:
            det = self.coco.loadRes(det_json_path)
        else:
            det = self.coco

        # 定义骨架连接关系（关节对）
        self.skeleton = [
            [0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7],
            [8, 9], [9, 10], [10, 11], [12, 13], [13, 14], [14, 15],
            [16, 17], [17, 18], [18, 19], [20, 21], [21, 22], [22, 23]
        ]

        self.valid_person_list = []
        obj_idx = 0
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = det.getAnnIds(imgIds=img_id)
            anns = det.loadAnns(ann_ids)
            for ann in anns:
                # only save person class
                if ann["category_id"] != 1:
                    print(f'warning: find not support id: {ann["category_id"]}, only support id: 1 (person)')
                    continue

                # COCO_val2017_detections_AP_H_56_person.json文件中只有det信息，没有keypoint信息，跳过检查
                if det_json_path is None:
                    # skip objs without keypoints annotation
                    if "keypoints" not in ann:
                        continue
                    if max(ann["keypoints"]) == 0:
                        continue

                xmin, ymin, w, h = ann['bbox']
                # Use only valid bounding boxes
                if w > 0 and h > 0:
                    info = {
                        "box": [xmin, ymin, w, h],
                        "image_path": os.path.join(self.img_root, img_info["file_name"]),
                        "image_id": img_id,
                        "image_width": img_info['width'],
                        "image_height": img_info['height'],
                        "obj_origin_hw": [h, w],
                        "obj_index": obj_idx,
                        "score": ann["score"] if "score" in ann else 1.
                    }

                    # COCO_val2017_detections_AP_H_56_person.json文件中只有det信息，没有keypoint信息，跳过
                    if det_json_path is None:
                        keypoints = np.array(ann["keypoints"]).reshape([-1, 3])
                        visible = keypoints[:, 2]
                        keypoints = keypoints[:, :2]
                        info["keypoints"] = keypoints
                        info["visible"] = visible
                        
                        # 计算腿长（关节间距离）
                        leg_lengths = self.calculate_leg_lengths(keypoints, visible)
                        info["leg_lengths"] = leg_lengths

                    self.valid_person_list.append(info)
                    obj_idx += 1
                    
    def calculate_leg_lengths(self, keypoints, visible):
        """计算关节对之间的距离作为腿长"""
        leg_lengths = np.zeros(len(self.skeleton), dtype=np.float32)
        
        for i, (joint1_idx, joint2_idx) in enumerate(self.skeleton):
            # 检查两个关节点是否都可见
            if (joint1_idx < len(visible) and joint2_idx < len(visible) and 
                visible[joint1_idx] > 0 and visible[joint2_idx] > 0):
                # 计算两点之间的欧氏距离
                joint1 = keypoints[joint1_idx]
                joint2 = keypoints[joint2_idx]
                distance = np.sqrt(np.sum((joint1 - joint2) ** 2))
                leg_lengths[i] = distance
            else:
                # 如果关节不可见，将距离设为0
                leg_lengths[i] = 0.0
                
        return leg_lengths

    def __getitem__(self, idx):
        target = copy.deepcopy(self.valid_person_list[idx])

        image = cv2.imread(target["image_path"])
        if image is None:
            raise FileNotFoundError(f"图像文件未找到或无法读取: {target['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image, person_info = self.transforms(image, target)
            
            # 如果经过变换后，需要重新计算腿长
            if "keypoints" in person_info and "visible" in person_info:
                # 将tensor转为numpy进行计算
                if isinstance(person_info["keypoints"], torch.Tensor):
                    keypoints_np = person_info["keypoints"].numpy()
                else:
                    keypoints_np = person_info["keypoints"]
                    
                if isinstance(person_info["visible"], torch.Tensor):
                    visible_np = person_info["visible"].numpy()
                else:
                    visible_np = person_info["visible"]
                
                # # 重新计算腿长
                # leg_lengths = self.calculate_leg_lengths(keypoints_np, visible_np)
                # person_info["leg_lengths"] = torch.tensor(leg_lengths, dtype=torch.float32)
            
            # 使用变换后的person_info替换target
            
            return image, person_info

        return image, target

    def __len__(self):
        return len(self.valid_person_list)

    @staticmethod
    def collate_fn(batch):
        imgs_tuple, targets_tuple = tuple(zip(*batch))
        imgs_tensor = torch.stack(imgs_tuple)
        return imgs_tensor, targets_tuple


if __name__ == '__main__':
    train = CocoKeypoint("E:/project/HRNet/data/ant2024/", dataset="val")
    print(len(train))
    t = train[0]
    print(t)
