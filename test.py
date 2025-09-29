import hashlib
import humanhash

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms

import os
import random
import numpy as np
import pandas as pd 
from tabulate import tabulate
from tqdm import tqdm
import sys
import subprocess
import argparse
from scipy.ndimage import gaussian_filter

from model import tips, omaly
from datasets import input_transforms, dataset, desc
from utils.metrics import image_level_metrics, pixel_level_metrics
from utils.visualize import visualizer
from utils.logger import get_logger, read_train_args

# def pixel_level_metrics(a,b,c):
    # return 0

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calc_score(vis_feat, txt_feat, temp):
    return F.softmax((vis_feat @ txt_feat.permute(0, 2, 1))/temp, dim=-1)

def regrid_upsample_smooth(flat_scores, size, sigma):
    h_w = int(flat_scores.shape[1] ** 0.5) 
    regrided = flat_scores.reshape(flat_scores.shape[0], h_w, h_w, -1).permute(0, 3, 1, 2)
    upsampled = torch.nn.functional.interpolate(regrided, (size, size), mode='bilinear').permute(0, 2, 3, 1)
    rough_maps = (1-upsampled[..., 0] + upsampled[..., 1])/2
    assert (rough_maps >= 0).all() and (rough_maps <= 1).all(), "All elements of rough_maps must be between 0 and 1"
    anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(map, sigma=sigma)) for map in rough_maps.detach().cpu()], dim=0)
    return anomaly_map

def test(args):
    logger = get_logger(args.save_path)
    # load dataset 
    image_size = 518 #224 if is_low_res else 448
    transform, target_transform = input_transforms.create_transforms(image_size)

    # load model
    device = 'cuda'
    tips_vision_encoder, tips_text_encoder, tokenizer, temperature = tips.load_model.get_model(args.models_dir, args.model_version)
    text_encoder = omaly.text_encoder(tokenizer, tips_text_encoder.to(device), 64, args.prompt_learn_method, args.fixed_prompt_type, args.n_prompt, args.n_deep_tokens, args.d_deep_tokens)
    vision_encoder = omaly.vision_encoder(tips_vision_encoder.to(device))

    # class_names = desc.dataset_dict[args.dataset]
    test_data = dataset.Dataset(args.data_path, transform, target_transform, args)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=True)
    fixed_class_names = [clss.replace('_', ' ') for clss in test_data.cls_names]
    
    # extract features
    with torch.no_grad():
        # Fixed prototypes
        fixed_text_features = text_encoder(fixed_class_names, device, learned=False)
        fixed_text_features = fixed_text_features / fixed_text_features.norm(dim=-1, keepdim=True) # NOTE: For test also 
    cls_text_features, seg_text_features = fixed_text_features, fixed_text_features
    
    if args.checkpoint_path:
        assert not args.prompt_learn_method == 'none', 'The prompt_learn_method should not be none'
        chekpoint = torch.load(args.params_path, weights_only=False)
        # text_encoder.learnable_prompts = chekpoint["learnable_prompts"]
        text_encoder.learnable_prompts = chekpoint
        # text_encoder.deep_parameters = chekpoint["deep_parameters"]
        
        learnable_class_names = ['object']
        learnable_class_ids = torch.tensor([0])
        print('The learnable prompts are read')
        
        # extract features
        with torch.no_grad():
            # Learnable prototypes
            learnable_text_features = text_encoder(learnable_class_names, device, learned=True) # NOTE: important learned=True
            learnable_text_features = learnable_text_features / learnable_text_features.norm(dim=-1, keepdim=True) # NOTE: For test also 
        cls_text_features, seg_text_features = learnable_text_features, learnable_text_features
    
    if args.checkpoint_path and args.decoupled_prompt:
        cls_text_features, seg_text_features = fixed_text_features, learnable_text_features
        
    dataset_preds = {cls_id: {'name': test_loader.dataset.cls_names[cls_id], 'img_scrs': [], 'img_lbls': [], 'pxl_scrs': [], 'pxl_lbls': [], 'paths': []} for cls_id in test_loader.dataset.class_ids}
    for batch in tqdm(test_loader, desc="Extracting features", unit="batch"):
        image = batch['img'].to(device)
        label = batch['anomaly'].long().to(device)
        abnorm_mask = batch['abnorm_mask'].squeeze(dim=1).to(device)
        path = batch['img_path']
        
        # Indecies
        cls_class_ids, seg_class_ids = batch['cls_id'], batch['cls_id']
        if args.checkpoint_path and args.decoupled_prompt:
            seg_class_ids = learnable_class_ids
        elif args.checkpoint_path and not args.decoupled_prompt:
            cls_class_ids, seg_class_ids = learnable_class_ids, learnable_class_ids

        with torch.no_grad():
            vision_features = vision_encoder(image)
            vision_features = [feature / feature.norm(dim=-1, keepdim=True) for feature in vision_features] # NOTE: for test also

            # calculate normal/abnormal scores
            img_scr0 = calc_score(vision_features[0], cls_text_features[cls_class_ids], temperature).squeeze(dim=1).detach() # prompt_class_ids cls_ids
            img_scr1 = calc_score(vision_features[1], cls_text_features[cls_class_ids], temperature).squeeze(dim=1).detach()
            
            img_map = calc_score(vision_features[2], seg_text_features[seg_class_ids], temperature)
            if args.aggregate_local2global:
                max_local = torch.max(img_map, dim=1)[0]
                img_scr0 = img_scr0 + max_local
                img_scr1 = img_scr1 + max_local
                
            pxl_scr = regrid_upsample_smooth(img_map.detach(), image_size, args.sigma)
            
        for idx, cls_id in enumerate(batch['cls_id'].cpu().numpy()):
            dataset_preds[cls_id]['img_scrs'].append([img_scr0[idx][1].cpu(), img_scr1[idx][1].cpu()])
            dataset_preds[cls_id]['img_lbls'].append(label[idx].cpu())
            dataset_preds[cls_id]['pxl_scrs'].append(pxl_scr[idx].cpu())
            dataset_preds[cls_id]['pxl_lbls'].append(abnorm_mask[idx].cpu())
            dataset_preds[cls_id]['paths'].append(path[idx])
        
    # calculate metrics
    header = ['objects']+args.pixel_metrics+[mtr for mtr in args.image_metrics for _ in range(2)]
    dataset_results = []
    for cls_id in dataset_preds.keys():
        cls_results = [dataset_preds[cls_id]['name']]
        img_prds = np.array(dataset_preds[cls_id]['img_scrs'])
        img_lbls = np.array(dataset_preds[cls_id]['img_lbls'])
        pxl_prds = torch.stack(dataset_preds[cls_id]['pxl_scrs'], dim=0)
        pxl_lbls = torch.stack(dataset_preds[cls_id]['pxl_lbls'], dim=0)
    
        for px_mtr in args.pixel_metrics:
            if not px_mtr == '':
                cls_results.append(pixel_level_metrics(pxl_prds, pxl_lbls, px_mtr)*100)
            
        for im_mtr in args.image_metrics:
            for col in range(img_prds.shape[1]):
                cls_results.append(image_level_metrics(img_prds[:, col], img_lbls, im_mtr)*100)
                
        if args.visualize:
            img_path = f"{args.dataset}/{dataset_preds[cls_id]['name']}"
            visualizer(dataset_preds[cls_id]['paths'], pxl_prds.cpu().numpy(), pxl_lbls.cpu().numpy(), image_size, img_path, save_path=f'{args.save_path}/img/', draw_contours=True)
            
        dataset_results.append(cls_results)
        
    df = pd.DataFrame(dataset_results, columns=header)
    mean_values = ['Mean'] + df.iloc[:, 1:].mean().tolist()
    # df = df.append(mean_values)
    df.loc[len(df)] = mean_values
    df = df.round(2)
    
    # store the results
    results_text = tabulate(df, headers='keys', tablefmt='pretty') 
    logger.info(results_text)
    
def make_human_readable_name(args, exclude=['model_name', 'dataset', 'dataset_category', 'data_path',
                                            'checkpoint_path', 'training_path', "Timestamp",
                                            "metrics", "devices", "epochs", "visualize", 'help', None]):
    args=vars(args)
    name_value_pairs = [
        f"{k}_{v}"
        for k,v in args.items()
        if k not in exclude # Exclude "help" or invalid arguments
    ]   
    combined = ",".join(sorted(name_value_pairs))  # Sorting ensures consistent order
    hash_value = hashlib.sha256(combined.encode()).hexdigest()
    human_hash = humanhash.humanize(hash_value, words=2) # 'hash' #
    return human_hash

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("TIPSomaly", add_help=True)
    # model
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--seed", type=int, default=111, help="random seed")

    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--devices", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7], help="array of possible cuda devices")
    parser.add_argument("--model_name", type=str, default="tips_test", help="")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="")
    parser.add_argument("--epoch", type=int, default=1, help="")

    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    
    parser.add_argument("--dataset", type=str, default="visa")
    parser.add_argument("--dataset_category", type=str, default='', help="train dataset categories")
    
    parser.add_argument("--class_name", type=str, nargs='+', default=['all'], help="train class name")
    
    parser.add_argument("--image_metrics", type=str, nargs='+', default=['auroc', 'ap', 'f1-max'], help="")
    parser.add_argument("--pixel_metrics", type=str, nargs='+', default=['auroc', 'aupro', 'f1-max'], help="")

    parser.add_argument("--k_shot", type=int, default=0, help="number of samples per class for few-shot learning. 0 means use all data.")
    
    parser.add_argument("--type", type=str, default='train') 
    parser.add_argument("--visualize", type=str2bool, default=False)
    parser.add_argument("--log_dir", type=str, default="")
    
    ##########################
    parser.add_argument("--model_version", type=str, default='l14h', choices=["s14h","b14h","l14h","so4h","g14l","g14h"])
    parser.add_argument("--models_dir", type=str, default='/home/alireza/.cache/tips/')
    
    parser.add_argument("--n_deep_tokens", type=int, default=0)
    parser.add_argument("--d_deep_tokens", type=int, default=0)
    parser.add_argument("--n_prompt", type=int, default=8)
    parser.add_argument("--fixed_prompt_type", type=str, default='industrial', choices=['industrial', 'medical_low', 'medical_high', 'object_agnostic'])
    
    parser.add_argument("--prompt_learn_method", type=str, default='concat', choices=['concat', 'sumate', 'entire_learnable', 'none'])
    parser.add_argument("--decoupled_prompt", type=str2bool, default=True)
    parser.add_argument("--aggregate_local2global", type=str2bool, default=True)
    
    args = parser.parse_args()
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.devices))
        command = [sys.executable, __file__, ] + sys.argv[1:] 
        process = subprocess.Popen(command, env=os.environ)
        process.wait()
        
    else:
        setup_seed(args.seed)

        args.data_path = [f'/data/alireza/datasets/{args.dataset_category}/{args.dataset}/']
        if not args.checkpoint_path:
            args.log_dir = make_human_readable_name(args)
            args.save_path = f'./workspaces/{args.model_name}/{args.log_dir}/quantative/NoTrain/{args.dataset}'
        else: # ./workspaces/test/blah-blah/checkpoints/
            splits = args.checkpoint_path.split('/')            
            args.params_path = f'{args.checkpoint_path}/learnable_params_{args.epoch}.pth'
            args.model_name = splits[-3]
            args.log_dir = splits[-2]
            args.save_path = f'{"/".join(splits[:-1])}/quantative/epoch_{args.epoch}/{args.dataset}'
            ### 
            train_args = read_train_args(args.checkpoint_path) # ./workspaces/{args.model_name}/{args.log_dir}/args.txt
            args.prompt_learn_method = train_args['prompt_learn_method']
            assert not train_args['prompt_learn_method'] is None, 'prompt_learn_method should not be none'
            
        print(args)
        print(f"Data Path: {args.data_path}, Log Directory: {args.log_dir}, Save Path: {args.save_path}")
        test(args) 