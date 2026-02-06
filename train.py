import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
import subprocess
import argparse
import hashlib
import humanhash
from collections import defaultdict
from tqdm import tqdm

import random
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel, AutoTokenizer, SiglipTextModel, SiglipVisionModel

from datasets import input_transforms, dataset
from utils.loss import FocalLoss, BinaryDiceLoss
from utils.logger import save_args_to_file, get_logger
from torch.utils.tensorboard import SummaryWriter

from model import tips
from model import omaly
from model.big_vision import load_siglip
from model.siglip2.siglip2_prompt_learnable import SiglipTextModelWithPromptLearning

loss_names = {'img_ls_ce': 'LS CE', 'pxl_ls_fc': 'LS FC', \
                'plx_ls_dc_p': 'LS DC P', 'plx_ls_dc_n': 'LS DC N', \
                'emb_l1_nrm': 'LS L1 NRM', 'epc_ls': 'total'}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = 111 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def calc_soft_score(vis_feat, txt_feat, temp):
    return F.softmax((vis_feat @ txt_feat.permute(0, 2, 1))/temp, dim=-1)

def calc_sigm_score(vis_feat, txt_feat, temp, bias):
    if vis_feat.dim() < 3:
        vis_feat = vis_feat.unsqueeze(dim=1)
    tempered_logits = vis_feat @ txt_feat.permute(0, 2, 1) * temp
    probs = 1 / (1 + np.exp(-tempered_logits - bias))
    return F.softmax(probs, dim=-1)

def calc_sigm_score_hf(vis_feat, txt_feat, temp_non_exp, bias):
    if vis_feat.dim() < 3:
        vis_feat = vis_feat.unsqueeze(dim=1)
    logits = vis_feat @ txt_feat.permute(0, 2, 1) * temp_non_exp.exp() + bias
    probs = torch.sigmoid(logits)
    return probs

def create_tips(args, device):
    # load dataset 
    transform, target_transform = input_transforms.create_transforms_tips(args.image_size)

    # load model
    vision_encoder, text_encoder, tokenizer, temperature = tips.load_model.get_model(args.models_dir, args.model_version)
    return vision_encoder.to(device), text_encoder.to(device), text_encoder.transformer.width, tokenizer, transform, target_transform, temperature

def create_siglip2(args, device):
    transform, target_transform = load_siglip.create_preprocessors_siglip2(args.image_size)
    vision_encoder, text_encoder, tokenizer = load_siglip.build_siglip_modules(args.model_version, args.image_size)
    # model.to(device)

    temperature, bias = text_encoder.params['t'], text_encoder.params['b']
    temperature = np.exp(torch.from_numpy(np.array(temperature)))
    return vision_encoder, text_encoder, text_encoder.model.out_dim[1], tokenizer, transform, target_transform, temperature, bias

def create_siglip2_hf(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_version)
    model = AutoModel.from_pretrained(args.model_version)
    text_encoder = SiglipTextModelWithPromptLearning.from_pretrained(args.model_version).to(device)
    vision_encoder = SiglipVisionModel.from_pretrained(args.model_version).to(device)
    processor = AutoProcessor.from_pretrained(args.model_version)
    def transform(x):
        d = processor(images=x, return_tensors="pt")
        return d['pixel_values'].squeeze(0)
    target_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    bias = model.logit_bias.to(device)
    temp_non_exp = model.logit_scale.to(device)
    return vision_encoder, text_encoder, model.text_model.embeddings.token_embedding.embedding_dim, tokenizer, transform, target_transform, temp_non_exp, bias

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
    # print(f"Parameters to be updated: {enabled}")

    model.eval()
    return model

def train(args):
    epochs = args.epoch
    device = args.device

    writer = SummaryWriter(log_dir=args.experiment_root)
    logger = get_logger(args.experiment_root)
    
    if args.backbone_name == 'tips':
        bb_vision_encoder, bb_text_encoder, text_embd_dim, tokenizer, transform, target_transform, temperature = create_tips(args, device)
        calc_score = lambda vis_feat, txt_feat: calc_soft_score(vis_feat, txt_feat, temperature)
    elif args.backbone_name == "siglip2":
        bb_vision_encoder, bb_text_encoder, text_embd_dim, tokenizer, transform, target_transform, temperature, bias  = create_siglip2(args, device)
        calc_score = lambda vis_feat, txt_feat: calc_sigm_score(vis_feat, txt_feat, temperature, bias)
    elif args.backbone_name == 'siglip2-hf':
        bb_vision_encoder, bb_text_encoder, text_embd_dim, tokenizer, transform, target_transform, temperature, bias = create_siglip2_hf(args, device)
        calc_score = lambda vis_feat, txt_feat: calc_sigm_score_hf(vis_feat, txt_feat, temperature, bias)

    bb_text_encoder = bb_text_encoder.to(device)
    bb_vision_encoder = bb_vision_encoder.to(device)
    bb_text_encoder = turn_gradient_off(bb_text_encoder)
    bb_vision_encoder = turn_gradient_off(bb_vision_encoder)
    text_encoder = omaly.text_encoder(tokenizer, bb_text_encoder, args.backbone_name, text_embd_dim, 64, args.prompt_learn_method, args.fixed_prompt_type, args.n_prompt, args.n_deep_tokens, args.d_deep_tokens)
    vision_encoder = omaly.vision_encoder(bb_vision_encoder, args.backbone_name)


    # load dataset 
    # class_names = desc.dataset_dict[args.dataset]
    train_data = dataset.Dataset(args.data_path, transform, target_transform, args)    

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=4, shuffle=False)

    # class_names = [clss.replace('_', ' ') for clss in train_data.cls_names]
    # class_ids = train_data.class_ids
    class_names = ['object']
    class_ids = torch.tensor([0])
    
    # Define losses 
    bce_loss = torch.nn.CrossEntropyLoss()
    loss_focal = FocalLoss() 
    loss_dice = BinaryDiceLoss()
    
    # Define optimizer
    optimizer = torch.optim.Adam(
        list(text_encoder.learnable_prompts),# + list(text_encoder.deep_parameters),
        lr=args.learning_rate,
        betas=(0.5, 0.999)
    )
    train_stats = defaultdict(list)

    torch.autograd.set_detect_anomaly(True)
    train_loader_cpu = [bat for bat in train_loader]
    
    global_step = 0

    text_encoder.train() 
    text_encoder.to(device)
    vision_encoder.train() 
    vision_encoder.to(device)
    for epoch in range(epochs):  # Add epoch loop
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = defaultdict(int)
        
        for batch in tqdm(train_loader_cpu, desc="Train", unit="batch"):
            image = batch['img'].to(device)
            # cls_ids = batch['cls_id']
            label = batch['anomaly'].long().to(device)
            abnorm_mask = batch['abnorm_mask'].squeeze(dim=1).to(device)
            
            # extract features
            text_features = text_encoder(class_names, device, learned=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) # NOTE: For test also 
            with torch.no_grad():
                vision_features = vision_encoder(image)
                vision_features = [feature / feature.norm(dim=-1, keepdim=True) for feature in vision_features] # NOTE: for test also
                
            # calculate normal/abnormal scores (since TIPS has two global visual embeddings we have two calculated image-level scores)
            img_scr0 = calc_score(vision_features[0], text_features[class_ids]).squeeze(dim=1)
            img_scr1 = calc_score(vision_features[1], text_features[class_ids]).squeeze(dim=1)
            
            img_map = calc_score(vision_features[2], text_features[class_ids])
            anomaly_map = regrid_upsample(img_map, args.image_size)
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

            # L1 Regularization term
            l1_norm = torch.sum(torch.abs(text_features))
            loss_total = loss_total + l1_norm * args.l1_lambda

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
            epoch_loss['emb_l1_nrm'] += l1_norm.item()

            # Tensorboard update for each batch
            writer.add_scalar(f"Loss/img_ls_ce", ls_cls.item(), global_step)
            writer.add_scalar(f"Loss/pxl_ls_fc", ls_fc.item(), global_step)
            writer.add_scalar(f"Loss/plx_ls_dc_p", ls_dc_p.item(), global_step)
            writer.add_scalar(f"Loss/plx_ls_dc_n", ls_dc_n.item(), global_step)
            writer.add_scalar(f"Loss/epc_ls", loss_total.item(), global_step)
            writer.add_scalar(f"Loss/emb_l1_nrm", l1_norm.item(), global_step)
            global_step += 1
        
        # Calc epoch mean loss
        num_batches = len(train_loader)
        for key, val in epoch_loss.items():
            train_stats[key].append(val / num_batches)
        
        # Print mean losses at the end of the epoch
        epoch_details = f"Epoch {epoch + 1} Mean Losses: "
        for key, val in epoch_loss.items():
            epoch_details = epoch_details + f"{loss_names[key]}: {train_stats[key][-1]:.4f}, " 
        logger.info(epoch_details[:-2])
        
        torch.save({"learnable_prompts":text_encoder.learnable_prompts}, 
                   f'{args.save_path}/learnable_params_{epoch+1}.pth')
                    # "deep_parameters":text_encoder.deep_parameters}, 
        print(f'checkpoints saved for epoch {epoch+1}.')
        
def make_human_readable_name(args, exclude=['model_name', 'dataset', 'dataset_category', 'epoch', 'data_path',
                                            'checkpoint_path', 'training_path', "Timestamp",
                                            "metrics", "device", "available_devices", "epochs", "visualize", 'help', None]):
    args=vars(args)
    name_value_pairs = [
        f"{k}_{v}"
        for k,v in args.items()
        if k not in exclude # Exclude "help" or invalid arguments
    ]   
    combined = ",".join(sorted(name_value_pairs))  # Sorting ensures consistent order
    hash_value = hashlib.sha256(combined.encode()).hexdigest()
    human_hash = humanhash.humanize(hash_value, words=2)
    return human_hash.replace('-', '_')

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

    parser.add_argument("--epoch", type=int, default=5, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001)

    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--device", type=str, default="cuda", help="type of device, can be cuda or cpu")
    parser.add_argument("--available_devices", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7], help="array of possible cuda devices")
    parser.add_argument("--model_name", type=str, default="tips_test", help="cuda device")
    parser.add_argument("--models_dir", type=str, default="./tips", help="directory of the base model of tips")
    parser.add_argument("--data_root_dir", type=str, default="./datasets", help="root directory for all datasets to be placed in")
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    
    parser.add_argument("--dataset", type=str, default="visa")
    parser.add_argument("--dataset_category", type=str, default='', help="train dataset categories")
    
    parser.add_argument("--type", type=str, default='train') 
    parser.add_argument("--class_name", type=str, nargs='+', default=['all'], help="train class name")
    parser.add_argument("--k_shot", type=int, default=0, help="number of samples per class for few-shot learning. 0 means use all data.")

    ##########################
    ### Method Arguements ####
    parser.add_argument("--model_version", type=str, default='l14h', choices=["s14h","b14h","l14h","so4h","g14l","g14h", \
                                                                                "B/16", "L/16", "So400m/14", "So400m/16", "g-opt/16", \
                                                                                    "google/siglip2-so400m-patch16-256", "google/siglip2-large-patch16-512"])
    parser.add_argument("--n_deep_tokens", type=int, default=0)
    parser.add_argument("--d_deep_tokens", type=int, default=0)
    parser.add_argument("--n_prompt", type=int, default=8)
    parser.add_argument("--fixed_prompt_type", type=str, default='industrial')
    
    parser.add_argument("--prompt_learn_method", type=str, default='concat', choices=['concat', 'sumate', 'entire_learnable', 'none'])
    parser.add_argument("--cls_seg_los", type=str, default='seg', choices=['both', 'seg', 'cls'])
    parser.add_argument("--l1_lambda", type=float, default=0.0)
    parser.add_argument("--backbone_name", type=str, default='tips', choices=["tips", "siglip2", "siglip2-hf"])

    args = parser.parse_args()        

    command = [sys.executable, __file__, ] + sys.argv[1:] 
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.available_devices)) if len(args.available_devices) > 1 else str(args.available_devices[0])
        process = subprocess.Popen(command, env=os.environ)
        process.wait()
        
    else:
        print(args)
        setup_seed(args.seed)
        args.log_dir = make_human_readable_name(args)
        args.data_path = [f'{args.data_root_dir}/{args.dataset_category}/{args.dataset}/']
        args.experiment_root = f'./workspaces/trained_on_{args.dataset}_{args.model_name}/{args.log_dir}'
        args.save_path = f'{args.experiment_root}/checkpoints'
        os.makedirs(args.save_path, exist_ok=True)
        
        save_args_to_file(args, command) # ./workspaces/{args.model_name}/{args.log_dir}/args.txt
        
        print(f"Data Path: {args.data_path}, Log Directory: {args.log_dir}, Save Path: {args.save_path}")
        train(args)