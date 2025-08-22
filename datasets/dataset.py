import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import os

def sample_data(meta_info, k_shot):
    sampled_data = []
    complement_data = [] 
    
    for cls_name, data_list in meta_info.items():
        nrm_smpls = [item for item in data_list if item['anomaly'] == 0]
        anm_smpls = [item for item in data_list if item['anomaly'] == 1]
        
        if k_shot > 0:
            n_samples = k_shot
            n_nrm_smpls = min(int(n_samples / 2), len(nrm_smpls))
            n_anm_smpls = min(int(n_samples / 2), len(anm_smpls))

        cls_data = []
        cls_data.extend(random.sample(nrm_smpls, n_nrm_smpls))
        cls_data.extend(random.sample(anm_smpls, n_anm_smpls))
        sampled_data.extend(cls_data)
        
        complement_class_data = [item for item in data_list if item not in cls_data]
        complement_data.extend(complement_class_data)
        
        print(f'num samples for cls {cls_name}, norm: {n_nrm_smpls}, anom: {n_anm_smpls}')
        
    return sampled_data, complement_data

class Dataset(data.Dataset):
    def __init__(self, roots, transform, target_transform, kwargs=None):
        self.roots = roots
        self.transform = transform
        self.target_transform = target_transform
        split='test'
        
        meta_infos = {}
        for root in roots:
            with open(f'{root}/meta.json', 'r') as f:
                meta_info = json.load(f)
                for cls in meta_info[split]:
                    meta_info[split][cls] = [{**s, 'root': root} for s in meta_info[split][cls]]
                    
                    if cls in meta_infos:
                        meta_infos[cls].extend(meta_info[split][cls])
                        meta_infos[cls].extend(meta_info[split][cls])
                    else:
                        meta_infos[cls] = meta_info[split][cls]
                        
        meta_info_classes = list(meta_infos.keys())

        self.selected_class = kwargs.class_name
        self.cls_names = meta_info_classes if self.selected_class == ['all'] else self.selected_class
        
        self.data_all = []
        for cls_name in self.cls_names:
            self.data_all.extend(meta_infos[cls_name])

        self.dataset_name = kwargs.dataset
        self.class_ids = list(range(len(self.cls_names)))
        self.class_name_map_class_id = {k: index for k, index in zip(self.cls_names, self.class_ids)}

        # Few-shot dataset (splitting...)
        self.k_shot = kwargs.k_shot
        if not self.k_shot == 0:
            sampled_sets = sample_data(meta_infos, self.k_shot)
            
            if kwargs.type == 'train':
                self.data_all = sampled_sets[0] 
            
            elif kwargs.type == 'test' and kwargs.train_dataset == self.dataset_name:
                self.data_all = sampled_sets[1] 
        
        self.length = len(self.data_all)
        print(f"number of train samples: {self.length}")
            
    def _process_image(self, data):
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], \
                                                                data['cls_name'], data['specie_name'], data['anomaly']
        
        root = data['root']    
        img = Image.open(os.path.join(root, img_path))

        if anomaly == 0 or (not os.path.isfile(os.path.join(root, mask_path))):
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        else:
            img_mask = np.array(Image.open(os.path.join(root, mask_path)).convert('L')) > 0
            img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(img_mask)
        img_mask = np.where(img_mask > 0.5, 1, 0)

        result = {
            'img': img,
            'abnorm_mask': img_mask,
            'cls_name': cls_name,
            'anomaly': anomaly,
            'img_path': os.path.join(root, img_path),
            "cls_id": self.class_name_map_class_id[cls_name]
        }

        return result

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        data = self.data_all[index]
        result = self._process_image(data)
        return result