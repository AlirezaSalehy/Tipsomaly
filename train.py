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

from model import tips
from model import omaly
from datasets import input_transforms, dataset, desc
from utils.metrics import image_level_metrics, pixel_level_metrics
from utils.loss import FocalLoss, BinaryDiceLoss
from utils.visualize import visualizer
from utils.logger import save_args_to_file, get_logger

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
    upsampled = regrid_upsample(flat_scores, size)
    anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(map, sigma=sigma)) for map in upsampled.detach().cpu()], dim=0)
    return anomaly_map

def regrid_upsample(flat_scores, size):
    h_w = int(flat_scores.shape[1] ** 0.5) 
    regrided = flat_scores.reshape(flat_scores.shape[0], h_w, h_w, -1).permute(0, 3, 1, 2)
    upsampled = torch.nn.functional.interpolate(regrided, (size, size), mode='bilinear').permute(0, 2, 3, 1)
    return upsampled

def turn_gradient_off(model):
    print("Turning off gradients in both the image and the text encoder")
    for _, param in model.named_parameters():
        param.requires_grad_(False)

    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")

    model.eval()
    return model

def evaluate(text_encoder, vision_encoder, data_loader, device, class_names, image_size, temperature):
    # Set to eval mode and disable gradient tracking
    text_encoder.eval()
    vision_encoder.eval()
    
    bce_loss = torch.nn.CrossEntropyLoss()
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    eval_stats = {'img_ls_ce': 0, 'pxl_ls_fc': 0, 'plx_ls_dc_p': 0, 'plx_ls_dc_n': 0, 'epc_ls': 0}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            image = batch['img'].to(device)
            cls_ids = batch['cls_id']
            label = batch['anomaly'].long().to(device)
            abnorm_mask = batch['abnorm_mask'].squeeze(dim=1).to(device)

            # Extract features
            text_features = text_encoder(class_names, device)
            vision_features = vision_encoder(image)

            # Calculate normal/abnormal scores
            img_scr0 = calc_score(vision_features[0], text_features[cls_ids], temperature).squeeze(dim=1)
            img_scr1 = calc_score(vision_features[1], text_features[cls_ids], temperature).squeeze(dim=1)

            img_map = calc_score(vision_features[2], text_features[cls_ids], temperature)
            anomaly_map = regrid_upsample(img_map, image_size)
            abnorm_mask[abnorm_mask > 0.5], abnorm_mask[abnorm_mask < 0.5] = 1, 0

            anomaly_map = anomaly_map.permute(0, 3, 1, 2)

            # Calculate losses
            ls_fc = loss_focal(anomaly_map, abnorm_mask)
            ls_dc_p = loss_dice(anomaly_map[:, 1, :, :], abnorm_mask)
            ls_dc_n = loss_dice(anomaly_map[:, 0, :, :], 1 - abnorm_mask)
            ls_ce = bce_loss(img_scr0, label) + bce_loss(img_scr1, label)

            loss_total = (ls_fc + ls_dc_p + ls_dc_n) + ls_ce

            # Accumulate losses
            eval_stats['img_ls_ce'] += ls_ce.item()
            eval_stats['pxl_ls_fc'] += ls_fc.item()
            eval_stats['plx_ls_dc_p'] += ls_dc_p.item()
            eval_stats['plx_ls_dc_n'] += ls_dc_n.item()
            eval_stats['epc_ls'] += loss_total.item()

    # Normalize by number of batches
    num_batches = len(data_loader)
    for k in eval_stats:
        eval_stats[k] /= num_batches

    return eval_stats

def test(text_encoder, vision_encoder, prompt_class_names, device, test_loader, prompt_class_ids, temperature, image_size, args):
    text_encoder.eval()
    vision_encoder.eval()
    
    with torch.no_grad():
        text_features = text_encoder(prompt_class_names, device)

    dataset_preds = {cls_id: {'name': test_loader.dataset.cls_names[cls_id], 'img_scrs': [], 'img_lbls': [], 'pxl_scrs': [], 'pxl_lbls': [], 'paths': []} for cls_id in test_loader.dataset.class_ids}
    for batch in tqdm(test_loader, desc="Extracting features", unit="batch"):
        image = batch['img'].to(device)
        cls_ids = batch['cls_id']
        label = batch['anomaly'].long().to(device)
        abnorm_mask = batch['abnorm_mask'].squeeze(dim=1).to(device)
        path = batch['img_path']
        
        with torch.no_grad():
            vision_features = vision_encoder(image)
            
            # calculate normal/abnormal scores
            img_scr0 = calc_score(vision_features[0], text_features[prompt_class_ids], temperature).squeeze(dim=1).detach()
            img_scr1 = calc_score(vision_features[1], text_features[prompt_class_ids], temperature).squeeze(dim=1).detach()
            
            img_map = calc_score(vision_features[2], text_features[prompt_class_ids], temperature)
            pxl_scr = regrid_upsample_smooth(img_map.detach(), image_size, args.sigma)
            pxl_scr = (1-pxl_scr[..., 0] + pxl_scr[..., 1])/2

        for idx, cls_id in enumerate(cls_ids.cpu().numpy()):
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
            cls_results.append(pixel_level_metrics(pxl_prds, pxl_lbls, px_mtr)*100)
            
        for im_mtr in args.image_metrics:
            for col in range(img_prds.shape[1]):
                cls_results.append(image_level_metrics(img_prds[:, col], img_lbls, im_mtr)*100)
                
        if True:
            img_path = f"{args.dataset}/{dataset_preds[cls_id]['name']}"
            visualizer(dataset_preds[cls_id]['paths'], pxl_prds.cpu().numpy(), pxl_lbls.cpu().numpy(), image_size, img_path, save_path=f'{args.save_path}/img/', draw_contours=True)
            
        dataset_results.append(cls_results)
        
    df = pd.DataFrame(dataset_results, columns=header)
    mean_values = ['Mean'] + df.iloc[:, 1:].mean().tolist()
    # df = df.append(mean_values)
    df.loc[len(df)] = mean_values
    df = df.round(2)
    
    # store the results
    results_path = f'{args.save_path}/results.txt'
    with open(results_path, 'w') as f:
        f.write(tabulate(df, headers='keys', tablefmt='pretty'))
    print(tabulate(df, headers='keys', tablefmt='pretty'))

# prompt_learn_method

def train(args):
    logger = get_logger(args.experiment_root)
    
    # load dataset 
    image_size = 448 #224 if is_low_res else 448
    learning_rate = 0.001
    epochs = 5
    transform, target_transform = input_transforms.create_transforms(image_size)

    # class_names = desc.dataset_dict[args.dataset]
    train_data = dataset.Dataset(args.data_path, transform, target_transform, args)    
    test_data = dataset.Dataset([f'/datasets/{args.dataset_category}/visa/'], transform, target_transform, args)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True,  
                                num_workers=8, pin_memory=True, prefetch_factor=2)    
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False,
                                num_workers=8, pin_memory=True, prefetch_factor=2)

    # class_names = [clss.replace('_', ' ') for clss in train_data.cls_names]
    # class_ids = train_data.class_ids
    class_names = ['object']
    class_ids = torch.tensor([0])
    
    # load model
    device = 'cuda'
    tips_vision_encoder, tips_text_encoder, tokenizer, temperature = tips.load_model.get_model('~/.cache/tips/', 'L', False)
    tips_text_encoder = turn_gradient_off(tips_text_encoder)
    tips_vision_encoder = turn_gradient_off(tips_vision_encoder)
    
    text_encoder = omaly.text_encoder(tokenizer, tips_text_encoder.to(device), 64, args.prompt_learn_method)
    vision_encoder = omaly.vision_encoder(tips_vision_encoder.to(device))
    
    # Define losses 
    bce_loss = torch.nn.CrossEntropyLoss()
    loss_focal = FocalLoss() 
    loss_dice = BinaryDiceLoss()
    
    # Define optimizer
    optimizer = torch.optim.Adam(text_encoder.learnable_parameters, lr=learning_rate, betas=(0.5, 0.999))  
    
    train_stats = {'img_ls_ce': [], 'pxl_ls_fc': [], 'plx_ls_dc_p': [], 'plx_ls_dc_n': [], 'epc_ls': []}
    torch.autograd.set_detect_anomaly(True)
    train_loader_cpu = [bat for bat in train_loader]
    
    # eval_stats = evaluate(text_encoder, vision_encoder, train_loader_cpu, device, class_names, image_size, temperature)  # Capture evaluation statistics
    # print(f"Evaluation Stats: {eval_stats}")  # Log evaluation statistics
    # args.visualize = False
    # test(text_encoder, vision_encoder, class_names, device, train_loader_cpu, class_ids, temperature, image_size, args)
    # args.visualize = True
        
    text_encoder.train() 
    vision_encoder.train() 
    for epoch in range(epochs):  # Add epoch loop
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = {'img_ls_ce': 0, 'pxl_ls_fc': 0, 'plx_ls_dc_p': 0, 'plx_ls_dc_n': 0, 'epc_ls': 0}
        
        for batch in tqdm(train_loader_cpu, desc="Train", unit="batch"):
            image = batch['img'].to(device)
            # cls_ids = batch['cls_id']
            label = batch['anomaly'].long().to(device)
            abnorm_mask = batch['abnorm_mask'].squeeze(dim=1).to(device)
            
            # extract features
            text_features = text_encoder(class_names, device)
            with torch.no_grad():
                vision_features = vision_encoder(image)
                
            # calculate normal/abnormal scores
            img_scr0 = calc_score(vision_features[0], text_features[class_ids], temperature).squeeze(dim=1)
            img_scr1 = calc_score(vision_features[1], text_features[class_ids], temperature).squeeze(dim=1)
            
            img_map = calc_score(vision_features[2], text_features[class_ids], temperature)
            anomaly_map = regrid_upsample(img_map, image_size)
            abnorm_mask[abnorm_mask > 0.5], abnorm_mask[abnorm_mask< 0.5] = 1, 0

            # Calculate loss
            anomaly_map = anomaly_map.permute(0, 3, 1, 2)
            ls_fc = loss_focal(anomaly_map, abnorm_mask)
            ls_dc_p = loss_dice(anomaly_map[:, 1, :, :], abnorm_mask)
            ls_dc_n = loss_dice(anomaly_map[:, 0, :, :], 1-abnorm_mask)
            
            ls_cls = bce_loss(img_scr0, label) + bce_loss(img_scr1, label) 
            ls_seg = ls_fc + ls_dc_p + ls_dc_n  # (pixel loss)
            if args.cls_seg_los == 'both':      # (image loss)
                loss_total = ls_cls + ls_seg
            elif args.cls_seg_los == 'seg':
                loss_total = ls_seg
            elif args.cls_seg_los == 'cls':
                loss_total = ls_cls

            # Train
            optimizer.zero_grad() 
            loss_total.backward() 
            optimizer.step()
            
            # log
            epoch_loss['img_ls_ce'] += ls_cls.item()
            epoch_loss['pxl_ls_fc'] += ls_fc.item()
            epoch_loss['plx_ls_dc_p'] += ls_dc_p.item()
            epoch_loss['plx_ls_dc_n'] += ls_dc_n.item()
            epoch_loss['epc_ls'] += loss_total.item()

            # Update tqdm description with current loss values
            tqdm.write(f"CE: {ls_cls.item():.4f}, FC: {ls_fc.item():.4f}, DC P: {ls_dc_p.item():.4f}, DC N: {ls_dc_n.item():.4f}")

        num_batches = len(train_loader)
        train_stats['img_ls_ce'].append(epoch_loss['img_ls_ce'] / num_batches)
        train_stats['pxl_ls_fc'].append(epoch_loss['pxl_ls_fc'] / num_batches)
        train_stats['plx_ls_dc_p'].append(epoch_loss['plx_ls_dc_p'] / num_batches)
        train_stats['plx_ls_dc_n'].append(epoch_loss['plx_ls_dc_n'] / num_batches)
        train_stats['epc_ls'].append(epoch_loss['epc_ls'] / num_batches)
        
        # Print mean losses at the end of the epoch
        epoch_details = f"Epoch {epoch + 1} Mean Losses: " +\
                        f"LS CE: {epoch_loss['img_ls_ce'] / num_batches:.4f}, " +\
                        f"LS FC: {epoch_loss['pxl_ls_fc'] / num_batches:.4f}, " +\
                        f"LS DC P: {epoch_loss['plx_ls_dc_p'] / num_batches:.4f}, " +\
                        f"LS DC N: {epoch_loss['plx_ls_dc_n'] / num_batches:.4f}, " +\
                        f" Loss: {epoch_loss['epc_ls'] / num_batches:.4f}"
        logger.info(epoch_details)
        
        torch.save(text_encoder.learnable_parameters, f'{args.save_path}/learnable_params_{epoch+1}.pth')
        print(f'checkpoints saved for epoch {epoch+1}.')
        
    # test(text_encoder, vision_encoder, class_names, device, test_loader, class_ids, temperature, image_size, args)
        

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
    human_hash = humanhash.humanize(hash_value, words=2)
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
    dss = ['mvtec']

    parser = argparse.ArgumentParser("TIPSomaly", add_help=True)
    # model
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--seed", type=int, default=111, help="random seed")

    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--devices", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7], help="array of possible cuda devices")
    parser.add_argument("--model_name", type=str, default="tips_test", help="cuda device")

    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    
    parser.add_argument("--dataset", type=str, nargs="+", default=[f'{ds}' for ds in dss], help="train dataset name")
    parser.add_argument("--dataset_category", type=str, default='', help="train dataset categories")
    
    parser.add_argument("--class_name", type=str, nargs='+', default=['all'], help="train class name")
    
    parser.add_argument("--k_shot", type=int, default=0, help="number of samples per class for few-shot learning. 0 means use all data.")
    
    parser.add_argument("--type", type=str, default='train') 
    parser.add_argument("--log_dir", type=str, default="")
    
    parser.add_argument("--image_metrics", type=str, nargs='+', default=['auroc', 'ap', 'f1-max'], help="")
    parser.add_argument("--pixel_metrics", type=str, nargs='+', default=['auroc', 'aupro', 'f1-max'], help="")
    
    parser.add_argument("--visualize", type=str2bool, default=True)

    ##########################

    parser.add_argument("--prompt_learn_method", type=str, default='concat', choices=['concat', 'sumate', 'entire_learnable', 'none'])
    parser.add_argument("--cls_seg_los", type=str, default='both', choices=['both', 'seg', 'cls'])

    args = parser.parse_args()        
    command = [sys.executable, __file__, ] + sys.argv[1:] 
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.devices)) if len(args.devices) > 1 else str(args.devices[0])
        process = subprocess.Popen(command, env=os.environ)
        process.wait()
        
    else:
        print(args)
        setup_seed(args.seed)
        args.log_dir = make_human_readable_name(args)
        args.data_path = [f'/datasets/{args.dataset_category}/{ds}/' for ds in args.dataset]
        args.experiment_root = f'./workspaces/trained_on_{"_".join(args.dataset)}_{args.model_name}/{args.log_dir}'
        args.save_path = f'{args.experiment_root}/checkpoints'
        os.makedirs(args.save_path, exist_ok=True)
        
        save_args_to_file(args, command) # ./workspaces/{args.model_name}/{args.log_dir}/args.txt
        
        print(f"Data Path: {args.data_path}, Log Directory: {args.log_dir}, Save Path: {args.save_path}")
        train(args)