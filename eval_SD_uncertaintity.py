import torch
#from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from modules.pipeline_stable_diffusion import StableDiffusionPipeline
#from torchmetrics.multimodal import CLIPScore
from PIL import Image
from datasets import load_dataset
from modules.unet_2D_conditioned import UNet2DConditionModel
from modules.scheduling_pndm import PNDMScheduler
from torchvision import transforms
import os
import torch.nn.functional as F
from eval_utils import *
from utils import *
from agg_experiments import get_artifact_mask, otsu_threshold, generate_special_batch_map_globalTimestep
import matplotlib.pyplot as plt
from config import set_config
from official_supervised_metrics import run_sup_metrics

from metrics_clipscore import calculate_clipscore_metric
from metrics import compute_fid_custom
from metrics2 import calculate_metrics

from artifacts_heatmap_generator.RichHF.model import  preprocess_image, RAHF
import argparse

NUM_SAMPLES_TO_GENERATE = 1000

PREMAID_DATASET = [
    
    
    "A golden retriever standing on green grass, holding a wooden stick gently in its mouth",
    "A young woman kneeling on the ground, planting a small sapling into soil with her hands",
    "A brown horse shaking its mane",
    "A tabby cat sitting on a flat rock, stretching one front paw forward",
    "A man standing barefoot on sand, throwing a smooth stone forward with one arm extended",
    "Two young wolves playfully nuzzling each other",
    "A child and a medium-sized dog facing each other as the child offers an open hand",
    "Two birds perched on the same low branch, one bird leaning toward the other as if chirping",
    "An adult sheep and a lamb walking side by side",
    "Two people standing on a dirt path in nature, exchanging a small object hand-to-hand",


    #"A white rabbit sitting on short grass, actively nibbling a small leaf",
    #"A middle-aged man pouring water from a metal can onto the ground",
    #"A young deer standing still, turning its head slightly as if listening",
    #"A woman sitting cross-legged on grass, closing her eyes while taking a deep breath",
    #"A black-and-white goat standing on a grassy patch, lowering its head to graze",
    #"Two horses standing close together, one gently nudging the other’s neck",
    #"An adult and a child walking slowly on grass, holding hands while looking at each other",
    #"Two cats sitting on the ground, one paw reaching out to touch the other",
    #"person and a horse standing side by side, the person brushing the horse’s mane",
    #"Two ducks floating on calm water, facing each other with their beaks nearly touching"



    #"A golden retriever mid-leap, catching a red frisbee in its mouth.",
    #"A snowy owl spreading its wings wide while taking flight from a snow-covered ground.",
    #"A young woman with long dark hair practicing yoga in a tree pose on a sandy beach",
    #"A red fox pouncing into deep snow to catch prey beneath the surface.",
    #"A chestnut horse galloping freely across an open prairie field." ,
    #"Two wolf cubs playfully wrestling in a forest clearing surrounded by autumn leaves.",
    #"A mother elephant gently touching her calf with her trunk on the African savanna.",
    #"Two hummingbirds hovering face-to-face near a cluster of red flowers in a garden setting.",
    #"A father and son sitting together on a wooden dock extending over a calm lake, fishing rods in hand. ",
    #"Two horses nuzzling each other nose-to-nose in a peaceful pasture at golden hour."
]


preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', type=str, default = 'demo', choices = ['generate_uncertaintity_samples', 
                                                                         'generate_eval_heatmaps', 
                                                                         'demo', 
                                                                         'compare_methods', 
                                                                         'manual_prepare',
                                                                         'analyze_compare_methods',
                                                                         'eval_var_uc',
                                                                         'qualitative',
                                                                         'playground',
                                                                         'eval_ablation',

                                                                         'calc_iou',
                                                                         'playground2',
                                                                         'motivation_exp_quant',
                                                                         'user_study',
                                                                         'motivation_exp'])
    parser.add_argument('--model', type=str, default = '1.5v', choices = ['1.5v', 'SDXL', 'PixArt'])


    parser.add_argument('--resize_fid', type=int, default = 299, choices = [299, 512, 1024])
    parser.add_argument('--compare_mode', type=str, default = "fid_filter_high", choices = ["fid_filter_high"])
    parser.add_argument('--compare_vis', action='store_true')
    parser.add_argument('--calc_clipscore', action='store_true')
    parser.add_argument('--calc_cmmd', action='store_true')
    parser.add_argument('--calc_sup_metrics', action='store_true')

    parser.add_argument('--calc_grad_fid', action='store_true')

    parser.add_argument('--partition', type=int)


    parser.add_argument('--demo_correct', action='store_true')

    parser.add_argument('--vis_score_dist', action='store_true')
    parser.add_argument('--use_global', action='store_true')

    parser.add_argument('--generate_var_uc_scores', action='store_true')


    


    parser.add_argument('--dataset', type=str, default = 'coco', choices = ['flickr8k', 'coco'])
    

    parser.add_argument('--generation_method', type=str, default = 'basic', choices = ['basic'])
    parser.add_argument('--agg_method', type=str, choices = ["sum", "max", "aboveAvg", "aboveOtsu"])

    parser.add_argument('--agg_MAD_method', type=str, choices = ["sum", "max", "count"])



    

    args = parser.parse_args()
    args.output_dir = f"uncertaintity_maps/{args.model}/{args.generation_method}/{args.dataset}"
    args.output_dir_demo = f"uncertaintity_maps_demo/{args.model}"
    args.output_dir_compare = f"uncertaintity_maps_compare/{args.dataset}/{args.model}/{args.compare_mode}/{args.resize_fid}"
    args.qualitative = f"uncertaintity_maps_demo/qualitative/{args.model}"
    args.user_study = f"uncertaintity_maps_demo/user_study/{args.model}"
    args.manual_prepare = f"uncertaintity_maps_demo/manual_prepare/{args.model}"




    args.output_vis_dir_compare = f"visualizations/compare/{args.dataset}/{args.model}/"





    if args.dataset == "coco":
        args.real_dataset_dir = "datasets/coco/val2014"

    args.batch_size = 16
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_demo, exist_ok=True)
    os.makedirs(args.output_dir_compare, exist_ok=True)
    os.makedirs(args.output_vis_dir_compare, exist_ok=True)
    os.makedirs(args.qualitative, exist_ok=True)
    os.makedirs(args.user_study, exist_ok=True)





    return args


def deterministic(seed) -> None:
    import numpy as np
    if seed is None:
        seed = 2024
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def demo_correct(args):
    import torchvision.transforms as T
    import matplotlib.cm as cm
    from diffusers import StableDiffusionInpaintPipeline 

    deterministic(2024)
    apply_var_method_uc = False

    seed = 0  # same seed as first generation
   
    for start_idx in range(0, 16, args.batch_size):
        dataset = load_dataset("jxie/flickr8k", split=f"validation[{start_idx}:{start_idx+args.batch_size}]", trust_remote_code=True)  # take 5 examples for demo
        
        
        prompts = [item["caption_0"] for item in dataset]
        generator = torch.Generator(device="cuda").manual_seed(seed)

        output = args.pipe(prompts, generator= generator, apply_uc = True, apply_uc_on_all_timesteps=True, return_mid_reps = True, apply_var_method_uc=apply_var_method_uc)

        images1 = output[0].images
        uncertainty_maps = output[1]["uncertainty_maps"]
        latents_lst = output[1]["latents_lst"]
        if apply_var_method_uc:
            pixel_wise_uncertainty_lst = output[1]["pixel_wise_uncertainty_lst"][9]

        '''for idx in range(len(images1)):
            print(prompts[idx])
            images1[idx].save(f"{args.output_dir_demo}/output{start_idx+idx}H1.jpg", quality=95)'''

        
        
        timesteps = sorted(uncertainty_maps.keys(), reverse=True)

    
        n_timesteps = len(timesteps)
        example_ts = timesteps[0]

        # Number of samples
        n_cols = len(uncertainty_maps[example_ts])#uncertainty_maps[example_ts][0].shape[0] // 2
        n_cols = 1 + n_cols
        last_layer_idx = n_cols - 2  # Last uncertainty map column
        num_samples = len(images1)
        masks = prepare_culumative_precentile(num_samples, last_layer_idx, uncertainty_maps, timesteps)

        generator = torch.Generator(device="cuda").manual_seed(seed)
        scheduler = DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler",torch_dtype=torch.float16,)
        new_model = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
         
            scheduler = scheduler,
        ).to("cuda")
        print(masks[0].shape)
        output = new_model(prompts, generator= generator, mask_image=torch.stack(masks,dim=0), scheduler=scheduler, strength=0.99, )

        images2 = output[0].images

        resize = T.Resize((256,256))

        for i in range(len(images1)):
            img1 = resize(images1[i])
            img2 = resize(images2[i])

            # mask -> convert to PIL (0-255), apply colormap
            mask = masks[i].squeeze()  # 1,64,64 -> 64,64
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)  # normalize
            mask = cm.hot(mask.cpu().numpy())[:, :, :3]  # apply 'hot' colormap, drop alpha
            mask = Image.fromarray((mask * 255).astype('uint8')).resize((256,256))

            # concat horizontally: [img1 | mask | img2]
            strip = Image.new("RGB", (256*3,256))
            strip.paste(img1, (0,0))
            strip.paste(mask, (256,0))
            strip.paste(img2, (512,0))

            strip.save(f"{args.output_dir_demo}/output{start_idx+i}H1.jpg", quality=95)
        exit(1)


       
        '''for idx in range(len(images)):
            print(prompts[idx])
            images[idx].save(f"{args.output_dir_demo}/output{start_idx+idx}H2.jpg", quality=95)
            if apply_var_method_uc:
                #summed = sum(pixel_wise_uncertainty_lst)
                #print(summed.shape)
                
                heatmap = stacked_pixel_wise_uncertainty_lst[idx] #summed.sum(dim=1)[0]
                plt.imshow(heatmap.detach().cpu().numpy(), cmap="hot")   # if you really need 'chot', use: cmap="hot"
                plt.axis('off')
                plt.savefig(f"{args.output_dir_demo}/output{start_idx+idx}H.jpg", bbox_inches='tight', pad_inches=0)
                plt.close()'''


        
        '''plot_uncertintiy_maps(
            uncertainty_maps, 
            images,
            prompts,
            out_dir = args.output_dir_demo,
            target_size=128,
            cmap="hot",
            start_idx = start_idx,
            culumative = True,
            dpi=150)'''
        
        




def demo(args):
    deterministic(2024)
    #deterministic(2025)
    apply_var_method_uc = False
   
    for start_idx in range(0, 16, args.batch_size):
        
        #if start_idx == 14:
        #    apply_var_method_uc = True

       
        dataset = load_dataset("jxie/flickr8k", split=f"validation[{start_idx}:{start_idx+args.batch_size}]", trust_remote_code=True)  # take 5 examples for demo
        
        
        prompts = [item["caption_0"] for item in dataset]
        prompts[0] = "Chef tossing pizza dough"
        prompts[1] = "Squirrel holding an acorn"

        
        #prompts[2] = 'Two people standing on a dirt path in nature, exchanging a small object hand-to-hand'
        #prompts[3] = 'An adult sheep and a lamb walking side by side'
        #prompts[4] = 'A brown horse shaking its mane'
        #prompts[5] = 'Two wolf cubs playfully wrestling in a forest clearing surrounded by autumn leaves.'
        #prompts[6] = 'Black dog jumping in the air with a stick in its mouth .'



        
            
        #ablation = False,
        output = args.pipe(prompts,  apply_uc = True, apply_uc_on_all_timesteps=True, ablation = True, return_mid_reps = True,  apply_var_method_uc=apply_var_method_uc)

        images = output[0].images
        uncertainty_maps = output[1]["uncertainty_maps"]
        latents_lst = output[1]["latents_lst"]
        #d_log = output[1]["d_log"]
        #for elem in d_log:
        #    print(elem)
        #exit(1)

        if apply_var_method_uc:
            pixel_wise_uncertainty_lst = output[1]["pixel_wise_uncertainty_lst"][9]

        
        #stacked_pixel_wise_uncertainty_lst = torch.stack(pixel_wise_uncertainty_lst, dim=1).sum(dim=1).sum(dim=1) 
       
        for idx in range(len(images)):
            print(prompts[idx])
            images[idx].save(f"{args.output_dir_demo}/output{start_idx+idx}.jpg", quality=95)
            if apply_var_method_uc:
                #summed = sum(pixel_wise_uncertainty_lst)
                #print(summed.shape)
                
                heatmap = stacked_pixel_wise_uncertainty_lst[idx] #summed.sum(dim=1)[0]
                plt.imshow(heatmap.detach().cpu().numpy(), cmap="hot")   # if you really need 'chot', use: cmap="hot"
                plt.axis('off')
                plt.savefig(f"{args.output_dir_demo}/output{start_idx+idx}H.jpg", bbox_inches='tight', pad_inches=0)
                plt.close()

        
        plot_uncertintiy_maps(
            uncertainty_maps, 
            images,
            prompts,
            out_dir = args.output_dir_demo,
            target_size=128,
            cmap="hot",
            start_idx = start_idx,
            culumative = True,
            dpi=150)
        #exit(1)
        exit(1)
        '''plot_ASCD(
            latents_lst, 
            images,
            prompts,
            uncertainty_maps,
            out_dir = f"{args.output_dir_demo}/ASCD",
            target_size=128,
            cmap="hot",
            start_idx = start_idx,
            dpi=150)'''

        '''plot_ASCD(
            latents_lst, 
            images,
            prompts,
            uncertainty_maps,
            out_dir = f"{args.output_dir_demo}/ASCD_OURS",
            target_size=128,
            cmap="hot",
            start_idx = start_idx,
            dpi=150,
            ours = True)'''
    

        #exit(1)


def qualitative(args):
    deterministic(2024)
    apply_var_method_uc = False

    modelRAHF = RAHF()
    ckpt_path = 'artifacts_heatmap_generator/RichHF/rahf_model.pt'
    modelRAHF.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    modelRAHF.eval()
    #args.batch_size = 1
    for start_idx in range(200, 400, args.batch_size):
        if start_idx == 16: #or start_idx == 14:
            apply_var_method_uc = True
        dataset = load_dataset("jxie/flickr8k", split=f"validation[{start_idx}:{start_idx+args.batch_size}]", trust_remote_code=True)  # take 5 examples for demo
        
        
        prompts = [item["caption_0"] for item in dataset]
    
        output = args.pipe(prompts, apply_uc = True, apply_uc_on_all_timesteps=True, return_mid_reps = True, apply_var_method_uc=apply_var_method_uc)

        images = output[0].images

        

        
        uncertainty_maps = output[1]["uncertainty_maps"]
        latents_lst = output[1]["latents_lst"]
        if apply_var_method_uc:
            pixel_wise_uncertainty_lst = output[1]["pixel_wise_uncertainty_lst"][-1]

        
            stacked_pixel_wise_uncertainty_lst = torch.stack(pixel_wise_uncertainty_lst, dim=1).sum(dim=1).sum(dim=1) 
       
        saved_image_paths = []
        for idx in range(len(images)):
            print(prompts[idx])
            images[idx].save(f"{args.qualitative}/output{start_idx+idx}.jpg", quality=95)
            saved_image_paths.append(f"{args.qualitative}/output{start_idx+idx}.jpg")
            if apply_var_method_uc:
                #summed = sum(pixel_wise_uncertainty_lst)
                #print(summed.shape)
                
                heatmap = stacked_pixel_wise_uncertainty_lst[idx] #summed.sum(dim=1)[0]

                save_overlay(
                    image_pil=images[idx],
                    heatmap=heatmap,
                    out_path=f"{args.qualitative}/output{start_idx+idx}_var_overlay.jpg",
                    alpha=0.75,
                    cmap="hot",
                    target_size = 512 if args.model == "1.5v" else 1024
                )

                #plt.imshow(heatmap.detach().cpu().numpy(), cmap="hot")   # if you really need 'chot', use: cmap="hot"
                #plt.axis('off')
                #plt.savefig(f"{args.qualitative}/output{start_idx+idx}_var.jpg", bbox_inches='tight', pad_inches=0)
                #plt.close()
        if start_idx == 16:
            image_rahf = torch.stack([preprocess_image(im) for im in saved_image_paths])
            outRAHF = modelRAHF(image_rahf.squeeze(1), prompts)
            heatmaps_batch = outRAHF.pop('heatmaps')
            heatmaps_batch = heatmaps_batch['implausibility']
            
            for idx in range(len(images)):
                save_overlay(
                    image_pil=images[idx],
                    heatmap=heatmaps_batch[idx].detach(),
                    out_path=f"{args.qualitative}/output{start_idx+idx}_sup_overlay.jpg",
                    alpha=0.75,
                    cmap="hot",
                    target_size = 512 if args.model == "1.5v" else 1024
                )
        
        timesteps = sorted(uncertainty_maps.keys(), reverse=True)

    
        n_timesteps = len(timesteps)
        example_ts = timesteps[0]

        # Number of samples
        n_cols = len(uncertainty_maps[example_ts])#uncertainty_maps[example_ts][0].shape[0] // 2
        n_cols = 1 + n_cols
        last_layer_idx = n_cols - 2  # Last uncertainty map column
        num_samples = len(images)
        masks = prepare_culumative_precentile(num_samples, last_layer_idx, uncertainty_maps, timesteps)

        for idx in range(len(images)):
            save_overlay(
                image_pil=images[idx],
                heatmap=masks[idx],
                out_path=f"{args.qualitative}/output{start_idx+idx}_cumperc_overlay.jpg",
                alpha=0.75,
                cmap="hot",
                target_size = 512 if args.model == "1.5v" else 1024
            )

        #print(len(masks))
        #print(masks[0].shape)

        if len(timesteps) == 50:
            start_idx_asced = 10
            end_idx_asced = 24
        else:
            start_idx_asced = 5
            end_idx_asced = 12
        ASCED_masks =   get_artifact_mask(
                            torch.stack(latents_lst[start_idx_asced:end_idx_asced], dim=1),
                            mad_scale= 3,
                            min_area = 4,
                            max_area = 5000,
                            min_width = 1,
                            expand_size = 0,
                            agg_type = "sum",
                            return_map = True
                        )
        for idx in range(len(images)):
            save_overlay(
                image_pil=images[idx],
                heatmap=ASCED_masks[idx],
                out_path=f"{args.qualitative}/output{start_idx+idx}_asced_overlay.jpg",
                alpha=0.75,
                cmap="hot",
                target_size = 512 if args.model == "1.5v" else 1024
            )
        #exit(1)     



        '''plot_uncertintiy_maps(
            uncertainty_maps, 
            images,
            prompts,
            out_dir = args.output_dir_demo,
            target_size=128,
            cmap="hot",
            start_idx = start_idx,
            culumative = True,
            dpi=150)'''
        
        '''plot_ASCD(
            latents_lst, 
            images,
            prompts,
            uncertainty_maps,
            out_dir = f"{args.output_dir_demo}/ASCD",
            target_size=128,
            cmap="hot",
            start_idx = start_idx,
            dpi=150)'''

        '''plot_ASCD(
            latents_lst, 
            images,
            prompts,
            uncertainty_maps,
            out_dir = f"{args.output_dir_demo}/ASCD_OURS",
            target_size=128,
            cmap="hot",
            start_idx = start_idx,
            dpi=150,
            ours = True)'''






def manual_prepare(args):
    deterministic(2024)
    
    args.apply_uc = True # False if args.generate_var_uc_scores else True, 
    args.apply_uc_on_all_timesteps= True # False if args.generate_var_uc_scores else True, 
    args.return_mid_reps = True # if args.generate_var_uc_scores else False
    dataset = dataset = load_dataset("jxie/flickr8k", split="validation", trust_remote_code=True).shuffle(seed=2021) #load_dataset("jxie/flickr8k", split=f"validation[:{NUM_SAMPLES_TO_GENERATE}]", trust_remote_code=True) 
    args.dataset = "flickr8k"
    
    flag_cant_resume = True

    start_iter = 0
    end_iter = 100
    sample_idx = 0
   
    for batch_start in range(start_iter, end_iter, args.batch_size):
       
         
        batch_end = min(batch_start + args.batch_size, len(dataset))
        batch_items = dataset[batch_start:batch_end]
        
        # Select shortest caption for each item in batch
        prompts = []

        if args.dataset == "flickr8k":
            for i in range(len(batch_items['caption_0'])):
                # Get all captions for this item
                captions = [batch_items[f'caption_{j}'][i] for j in range(5) if f'caption_{j}' in batch_items]
                # Select shortest
                shortest_caption = min(captions, key=len)
                prompts.append(shortest_caption)
        elif args.dataset == "coco":
            prompts = batch_items['caption']
            
        # Generate images
        output = args.pipe(prompts, apply_uc = args.apply_uc, apply_uc_on_all_timesteps=args.apply_uc_on_all_timesteps, 
                            return_mid_reps = args.return_mid_reps, return_aux = True,
                            apply_var_method_uc= args.generate_var_uc_scores)

        
        images = output[0].images
        uncertainty_maps = output[1]["uncertainty_maps"]
        latents_lst = output[1]["latents_lst"]
        pixel_wise_uncertainty_lst = output[1]["pixel_wise_uncertainty_lst"]

        aux_list = output[1]["aux_list"]

        print(len(aux_list))
        print(len(latents_lst))

       
        # Save each image with its prompt
        sample_idx_copy = sample_idx
        for idx in range(len(images)):
            # Create subdirectory for this sample
            sample_dir = os.path.join(args.manual_prepare, str(sample_idx))
            os.makedirs(sample_dir, exist_ok=True)

            # Save prompt to txt file
            with open(os.path.join(sample_dir, "prompt.txt"), "w") as f:
                f.write(prompts[idx])
            
            # Save image
            images[idx].save(os.path.join(sample_dir, "output.jpg"), quality=95)
            
            print(f"Sample {sample_idx}: {prompts[idx]}")
            sample_idx += 1
  
        if True: #args.generate_var_uc_scores == False:
            save_uncertainty_maps(
                uncertainty_maps, 
                sample_idx_copy,
                aux_list,
                out_dir = args.manual_prepare,
                cmap    = "hot",
                dpi=150,
            )
        #exit(1)
        #exit(1)

        


def user_study(args):
    import pandas as pd
    from datasets import Dataset

    deterministic(2025)
    op_captions = 'flicker'
    modelRAHF = RAHF()
    ckpt_path = 'artifacts_heatmap_generator/RichHF/rahf_model.pt'
    modelRAHF.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    modelRAHF.eval()

    csv_path = f"{args.user_study}/mapping.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["sample_id", "A", "B", "C"])
    csv_writer.writeheader()

    if op_captions == "coco":
        coco_dir = "datasets/coco"
        data_file = f'{coco_dir}/annotations/captions_val2014.json'
        data = json.load(open(data_file))

        # merge images and annotations

        images = data['images']
        
        annotations = data['annotations']
        df = pd.DataFrame(images)
        df_annotations = pd.DataFrame(annotations)
        df = df.merge(pd.DataFrame(annotations), how='left', left_on='id', right_on='image_id')

        # keep only the relevant columns
        df = df[['file_name', 'caption']]

        # remove duplicate images
        df = df.drop_duplicates(subset='file_name')
        
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        dataset = dataset.select(range(30000))

    elif op_captions == "flicker":
        dataset = load_dataset("jxie/flickr8k", split="validation", trust_remote_code=True) 
    
    else:
        dataset = PREMAID_DATASET
    if op_captions:
        all_dataset = dataset.shuffle(seed=2021)
    for start_idx in range(0, 10, args.batch_size):
        if op_captions:
            dataset = all_dataset.select(
                                    range(start_idx, start_idx + args.batch_size)
                                )  # take 5 examples for demo
                                        
            if op_captions == "coco":
                prompts = [item["caption"] for item in dataset]
            elif op_captions == "flicker":
                prompts = [item["caption_0"] for item in dataset]
        else:
            prompts = PREMAID_DATASET[start_idx: start_idx + args.batch_size]

    
        output = args.pipe(prompts,  apply_uc = True, apply_uc_on_all_timesteps=True, return_mid_reps = True)

        images = output[0].images

        uncertainty_maps = output[1]["uncertainty_maps"]
        latents_lst = output[1]["latents_lst"]
       
        saved_image_paths = []
        for idx in range(len(images)):
            print(prompts[idx])
            images[idx].save(f"{args.user_study}/output{start_idx+idx}.jpg", quality=95)
            saved_image_paths.append(f"{args.user_study}/output{start_idx+idx}.jpg")
            

        #image_rahf = torch.stack([preprocess_image(im) for im in saved_image_paths])
        #outRAHF = modelRAHF(image_rahf.squeeze(1), prompts)
        #heatmaps_batch = outRAHF.pop('heatmaps')
        #heatmaps_batch = heatmaps_batch['implausibility']
        
        asced_start_step = 10 if args.model != "PixArt" else 5
        asced_start_end = 24 if args.model != "PixArt" else 12
        
        
        timesteps = sorted(uncertainty_maps.keys(), reverse=True)
        n_timesteps = len(timesteps)
        example_ts = timesteps[0]

        # Number of samples
        n_cols = len(uncertainty_maps[example_ts])#uncertainty_maps[example_ts][0].shape[0] // 2
        n_cols = 1 + n_cols
        last_layer_idx = n_cols - 2  # Last uncertainty map column
        num_samples = len(images)
        masks = prepare_culumative_precentile(num_samples, last_layer_idx, uncertainty_maps, timesteps)
        heatmaps_batch =  masks
        '''
        ASCED_masks =   get_artifact_mask(
                            torch.stack(latents_lst[asced_start_step:asced_start_end], dim=1),
                            mad_scale= 3,
                            min_area = 4,
                            max_area = 5000,
                            min_width = 1,
                            expand_size = 0,
                            agg_type = "sum",
                            return_map = True
                        )'''
       
        

        for idx in range(len(images)):
            ref_img = images[idx].resize(
                (512 if args.model == "1.5v" else 1024,) * 2
            )

            

            overlays = {
                "sup": overlay_to_pil_simple(
                    images[idx],
                    heatmaps_batch[idx].detach(),
                    target_size=ref_img.size[0],
                ),
                "cumperc": overlay_to_pil_simple(
                    images[idx],
                    masks[idx],
                    target_size=ref_img.size[0],
                ),
                #"asced": overlay_to_pil_simple(
                #    images[idx],
                #    ASCED_masks[idx],
                #    target_size=ref_img.size[0],
                #),
            }

            compose_row(
                reference=ref_img,
                overlays_dict=overlays,
                out_path=f"{args.user_study}/sample_{start_idx+idx}.jpg",
                sample_id=start_idx + idx,
                csv_writer=csv_writer,
            )
             




def motivation_exp(args):
    deterministic(2024)
    
    for start_idx in range(0, 500, args.batch_size):
        dataset = load_dataset("jxie/flickr8k", split=f"validation[{start_idx}:{start_idx+args.batch_size}]", trust_remote_code=True)  # take 5 examples for demo
        
        prompts = [item["caption_0"] for item in dataset]

        generator = torch.Generator(device="cuda").manual_seed(2024)
        output = args.pipe(prompts, generator = generator, apply_uc = True, return_aux = True, apply_uc_on_all_timesteps=True, return_mid_reps = True, apply_var_method_uc=False)

        images = output[0].images
        uncertainty_maps = output[1]["uncertainty_maps"]
        aux_list         = output[1]["aux_list"]

        for idx in range(len(images)):
            curr_idx = start_idx+idx
            images[idx].save(f"tmp_motivation/output{curr_idx}.jpg", quality=95)
            update_json2(f"tmp_motivation/res.json", {curr_idx : {"prompt": prompts[idx]} })
   

        timesteps = sorted(uncertainty_maps.keys(), reverse=True)
   
        example_ts = timesteps[0]
        last_layer = sorted(uncertainty_maps[example_ts].keys())[-1]

        num_samples = int(uncertainty_maps[example_ts][last_layer].shape[0] / 2)
        
        for heatmap_idx in range(num_samples):
            curr_idx = str(start_idx+heatmap_idx)
            for map_idx in range(1):
                for row_idx, ts in enumerate(timesteps):
                    uncertainty = uncertainty_maps[ts][last_layer].chunk(2)[map_idx][heatmap_idx].squeeze(0)
                    latent = aux_list[row_idx][heatmap_idx]

                    update_json2(f"tmp_motivation/res.json", {curr_idx : {"score": {ts: float(uncertainty.sum().item())}} }, debug=False)

                    update_json2(f"tmp_motivation/res.json", {curr_idx : {"score_aux": {ts: float(latent.sum().item())}} })
        
    
    '''score_results_good, score_results_bad, images_good, images_bad = split_lists(score_results, all_images)
    score_results_good, score_results_bad, prompts_good, prompts_bad = split_lists(score_results, general_prompts)
        
    good_dir = f"tmp_motivation/good"
    os.makedirs(good_dir,exist_ok = True)
    for idx in range(len(score_results_good)):
        curr_dir = f"{good_dir}/{str(idx).zfill(4)}"
        os.makedirs(curr_dir, exist_ok = True)
         
        
        images_good[idx].save(f"{curr_dir}/output.jpg", quality=95)
        with open(os.path.join(curr_dir, "prompt.txt"), "w") as f:
            f.write(prompts_good[idx])
            
    bad_dir = f"tmp_motivation/bad"
    os.makedirs(bad_dir,exist_ok = True)
    for idx in range(len(score_results_bad)):
        curr_dir = f"{bad_dir}/{str(idx).zfill(4)}"
        os.makedirs(curr_dir, exist_ok = True)
         
        images_bad[idx].save(f"{curr_dir}/output.jpg", quality=95)
        with open(os.path.join(curr_dir, "prompt.txt"), "w") as f:
            f.write(prompts_bad[idx])   '''

    



def motivation_exp_quant(args):
    deterministic(2024)
    modelRAHF = RAHF()
    ckpt_path = 'artifacts_heatmap_generator/RichHF/rahf_model.pt'
    modelRAHF.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    modelRAHF.eval()
    
    
    for start_idx in range(0, 500, args.batch_size):
        dataset = load_dataset("jxie/flickr8k", split=f"validation[{start_idx}:{start_idx+args.batch_size}]", trust_remote_code=True)
        
        prompts = [item["caption_0"] for item in dataset]

        generator = torch.Generator(device="cuda").manual_seed(2024)
        output = args.pipe(prompts, generator=generator, apply_uc=True, return_aux=True, apply_uc_on_all_timesteps=True, return_mid_reps=True, apply_var_method_uc=False)

        images = output[0].images
        uncertainty_maps = output[1]["uncertainty_maps"]
        aux_list = output[1]["aux_list"]
        
        saved_image_paths = []
        for idx in range(len(images)):
            curr_idx = start_idx + idx
            images[idx].save(f"tmp_motivation_exp/output{curr_idx}.jpg", quality=95)
            saved_image_paths.append(f"tmp_motivation_exp/output{curr_idx}.jpg")
            update_json2(f"tmp_motivation_exp/res.json", {curr_idx: {"prompt": prompts[idx]}})
        
        image_rahf = torch.stack([preprocess_image(im) for im in saved_image_paths])
        outRAHF = modelRAHF(image_rahf.squeeze(1), prompts)
        heatmaps_batch = outRAHF.pop('heatmaps')
        heatmaps_batch = heatmaps_batch['implausibility'].detach().cpu()  # Shape: [batch_size, 384, 384]

        #print(heatmaps_batch.shape)
   
        timesteps = sorted(uncertainty_maps.keys(), reverse=True)
        example_ts = timesteps[0]
        last_layer = sorted(uncertainty_maps[example_ts].keys())[-1]
        num_samples = int(uncertainty_maps[example_ts][last_layer].shape[0] / 2)
        
        target_size = heatmaps_batch.shape[1:]  # (384, 384)
        
        for heatmap_idx in range(num_samples):
            curr_idx = str(start_idx + heatmap_idx)
            gt_heatmap = heatmaps_batch[heatmap_idx]  # Shape: [384, 384]
            
            gt_heatmap = (gt_heatmap - gt_heatmap.min()) / (gt_heatmap.max() - gt_heatmap.min())
            # Binarize GT heatmap using both methods
            gt_mean_binary = (gt_heatmap > gt_heatmap.mean()).float()
            gt_percentile_binary = (gt_heatmap > torch.quantile(gt_heatmap, 0.9)).float()

            gt_otsu_binary = (gt_heatmap > otsu_threshold(gt_heatmap.cpu().detach().numpy())).float()

            

            
            for map_idx in range(1):
                for row_idx, ts in enumerate(timesteps):
                    uncertainty = uncertainty_maps[ts][last_layer].chunk(2)[map_idx][heatmap_idx].squeeze(0).detach().cpu()  # [64, 64]
                    latent = aux_list[row_idx][heatmap_idx].detach().cpu()  # [64, 64]
                    
                    # Resize maps to match GT heatmap size
                    uncertainty_resized = F.interpolate(
                        uncertainty.unsqueeze(0).unsqueeze(0), 
                        size=target_size, 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0).squeeze(0)  # [384, 384]
                    
                    latent_resized = F.interpolate(
                        latent.unsqueeze(0).unsqueeze(0), 
                        size=target_size, 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0).squeeze(0)  # [384, 384]
                    
                    uncertainty_resized = (uncertainty_resized - uncertainty_resized.min()) / (uncertainty_resized.max() - uncertainty_resized.min())
                    latent_resized = (latent_resized - latent_resized.min()) / (latent_resized.max() - latent_resized.min())

                    # Calculate binary maps using mean threshold
                    uncertainty_mean_binary = (uncertainty_resized > uncertainty_resized.mean()).float()
                    latent_mean_binary = (latent_resized > latent_resized.mean()).float()
                    
                    
                    # Calculate binary maps using 0.5 percentile threshold
                    uncertainty_percentile_binary = (uncertainty_resized > torch.quantile(uncertainty_resized, 0.9)).float()
                    latent_percentile_binary = (latent_resized > torch.quantile(latent_resized, 0.9)).float()
                    
                    
                    uncertainty_otsu_binary = (uncertainty_resized > otsu_threshold(uncertainty_resized.cpu().detach().numpy())).float()
                    latent_otsu_binary = (latent_resized > otsu_threshold(latent_resized.cpu().detach().numpy())).float()

                    
                    # Calculate IoU scores
                    def calculate_iou(pred_binary, gt_binary):
                        intersection = (pred_binary * gt_binary).sum()
                        union = ((pred_binary + gt_binary) > 0).float().sum()
                        iou = (intersection / (union + 1e-8)).item()
                        return iou, (intersection / (gt_binary.sum() + 1e-8)).float().item()
                    
                    # IoU for uncertainty with mean threshold
                    iou_uncertainty_mean, intersection_uncertainty_mean = calculate_iou(uncertainty_mean_binary, gt_mean_binary)
                    update_json2(f"tmp_motivation_exp/res.json", 
                                {curr_idx: {"iou_uncertainty_mean": {ts: iou_uncertainty_mean}}})

                    update_json2(f"tmp_motivation_exp/res.json", 
                                {curr_idx: {"intersection_uncertainty_mean": {ts: intersection_uncertainty_mean}}})


                    iou_uncertainty_otsu, intersection_uncertainty_otsu = calculate_iou(uncertainty_otsu_binary, gt_otsu_binary)
                    update_json2(f"tmp_motivation_exp/res.json", 
                                {curr_idx: {"iou_uncertainty_otsu": {ts: iou_uncertainty_otsu}}})

                    update_json2(f"tmp_motivation_exp/res.json", 
                                {curr_idx: {"intersection_uncertainty_otsu": {ts: intersection_uncertainty_otsu}}})
                    
                    # IoU for uncertainty with percentile threshold
                    iou_uncertainty_percentile, intersection_uncertainty_percentile = calculate_iou(uncertainty_percentile_binary, gt_percentile_binary)
                    update_json2(f"tmp_motivation_exp/res.json", 
                                {curr_idx: {"iou_uncertainty_percentile": {ts: iou_uncertainty_percentile}}})
                    update_json2(f"tmp_motivation_exp/res.json", 
                                {curr_idx: {"intersection_uncertainty_percentile": {ts: intersection_uncertainty_percentile}}})
                    
                    # IoU for latent with mean threshold
                    iou_latent_mean, intersection_latent_mean = calculate_iou(latent_mean_binary, gt_mean_binary)
                    update_json2(f"tmp_motivation_exp/res.json", 
                                {curr_idx: {"iou_latent_mean": {ts: iou_latent_mean}}})

                    update_json2(f"tmp_motivation_exp/res.json", 
                                {curr_idx: {"intersection_latent_mean": {ts: intersection_latent_mean}}})
                    
                    # IoU for latent with percentile threshold
                    iou_latent_percentile, intersection_latent_percentile = calculate_iou(latent_percentile_binary, gt_percentile_binary)
                    update_json2(f"tmp_motivation_exp/res.json", 
                                {curr_idx: {"iou_latent_percentile": {ts: iou_latent_percentile}}})

                    update_json2(f"tmp_motivation_exp/res.json", 
                                {curr_idx: {"intersection_latent_percentile": {ts: intersection_latent_percentile}}})

                    
                    iou_latent_otsu, intersection_latent_otsu = calculate_iou(latent_otsu_binary, gt_otsu_binary)
                    update_json2(f"tmp_motivation_exp/res.json", 
                                {curr_idx: {"iou_latent_otsu": {ts: iou_latent_otsu}}})

                    update_json2(f"tmp_motivation_exp/res.json", 
                                {curr_idx: {"intersection_latent_otsu": {ts: intersection_latent_otsu}}})
                    
                    print(f"Sample {curr_idx}, Timestep {ts}:")
                    print(f"  Uncertainty Mean IoU: {iou_uncertainty_mean:.4f}")
                    print(f"  Uncertainty Percentile IoU: {iou_uncertainty_percentile:.4f}")
                    print(f"  Latent Mean IoU: {iou_latent_mean:.4f}")
                    
       


def compare_methods(args):
    all_unmaps = []
    all_latents = []

    time_steps_sorted = []
    # Iterate over subdirectories sorted numerically
    subdirs = sorted([d for d in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, d))], 
                    key=lambda x: int(x))
    
    #subdirs = subdirs[:10000]
    images_path = []
    for idx, subdir in enumerate(subdirs):

        subdir_path = os.path.join(args.output_dir, subdir)
        images_path.append(f"{subdir_path}/output.jpg")
        # Get all unmap files
        unmap_files = [f for f in os.listdir(subdir_path) if f.endswith("_unmap.pt")]
        # Sort descending by ts (numeric)
        unmap_files.sort(key=lambda x: int(x.split("_")[0]), reverse=True)

        time_steps_sorted = [int(elem.split("_")[0]) for elem in  unmap_files]
        
        # Load torch tensors
        unmaps = [os.path.join(subdir_path, f) for f in unmap_files]
        all_unmaps.append(unmaps)
        
        # Get all latent.py files
        latent_files = [f for f in os.listdir(subdir_path) if f.endswith("_latent.pt")]
        latent_files.sort(key=lambda x: int(x.split("_")[0]), reverse=True)

        latents = [os.path.join(subdir_path, f) for f in latent_files]
        all_latents.append(latents)
    
    
    dirs_dict = {
        "output_dir_compare" : args.output_dir_compare,
        "real_dataset_dir": args.real_dataset_dir,
        "fake_dataset_dir": args.output_dir,
        "compare_vis_dir": args.output_vis_dir_compare

    }

    '''print(all_unmaps[2])
    latents_to_load = [torch.load(elem) for elem in all_unmaps[2]]
    print(latents_to_load[0].shape)
    import matplotlib.pyplot as plt
    plt.imshow(latents_to_load[3].cpu(), cmap='hot')
    plt.colorbar()
    plt.savefig("tmp/heatmap.png", dpi=150)
    plt.close()
    exit(1)'''



    if args.vis_score_dist:
         eval_metrics_for_methods((all_unmaps, all_latents, time_steps_sorted), 
                                    args.methods_eval, 
                                    compare_mode = args.compare_mode, 
                                    dirs_dict   = dirs_dict ,
                                    resize_fid   = args.resize_fid,
                                    calc_clipscore = args.calc_clipscore,
                                    vis_score_dist = True
                                    )       

    
    elif args.compare_vis:
        vis_metrics_for_methods((all_unmaps, all_latents, time_steps_sorted), 
                                args.methods_eval, 
                                compare_mode = args.compare_mode, 
                                dirs_dict   = dirs_dict ,
                                resize_fid   = args.resize_fid,
                                images_path = images_path,
                                backup_best_worst = True,
                                jump_to_vis = True,
                                )


    else:
        eval_metrics_for_methods((all_unmaps, all_latents, time_steps_sorted), 
                                    args.methods_eval, 
                                    compare_mode = args.compare_mode, 
                                    dirs_dict   = dirs_dict ,
                                    resize_fid   = args.resize_fid,
                                    calc_clipscore = args.calc_clipscore,
                                    calc_cmmd =  args.calc_cmmd,
                                    calc_sup_metrics = args.calc_sup_metrics,
                                    calc_grad_fid = args.calc_grad_fid
                                    )
    












def compare_methods_tmp(args):
    all_unmaps = []
    all_latents = []

    time_steps_sorted = []
    # Iterate over subdirectories sorted numerically
    subdirs = sorted([d for d in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, d))], 
                    key=lambda x: int(x))
    
    #subdirs = subdirs[:100]
    images_path = []
    for idx, subdir in enumerate(subdirs):

        subdir_path = os.path.join(args.output_dir, subdir)
        images_path.append(f"{subdir_path}/output.jpg")
        # Get all unmap files
        unmap_files = [f for f in os.listdir(subdir_path) if f.endswith("_unmap.pt")]
        # Sort descending by ts (numeric)
        unmap_files.sort(key=lambda x: int(x.split("_")[0]), reverse=True)

        time_steps_sorted = [int(elem.split("_")[0]) for elem in  unmap_files]
        
        # Load torch tensors
        unmaps = [os.path.join(subdir_path, f) for f in unmap_files]
        all_unmaps.append(unmaps)
        
        # Get all latent.py files
        latent_files = [f for f in os.listdir(subdir_path) if f.endswith("_latent.pt")]
        latent_files.sort(key=lambda x: int(x.split("_")[0]), reverse=True)

        latents = [os.path.join(subdir_path, f) for f in latent_files]
        all_latents.append(latents)
    
    
    dirs_dict = {
        "output_dir_compare" : args.output_dir_compare,
        "real_dataset_dir": args.real_dataset_dir,
        "fake_dataset_dir": args.output_dir,
        "compare_vis_dir": args.output_vis_dir_compare

    }

    '''print(all_unmaps[2])
    latents_to_load = [torch.load(elem) for elem in all_unmaps[2]]
    print(latents_to_load[0].shape)
    import matplotlib.pyplot as plt
    plt.imshow(latents_to_load[3].cpu(), cmap='hot')
    plt.colorbar()
    plt.savefig("tmp/heatmap.png", dpi=150)
    plt.close()
    exit(1)'''



    if args.vis_score_dist:
         eval_metrics_for_methods((all_unmaps, all_latents, time_steps_sorted), 
                                    args.methods_eval, 
                                    compare_mode = args.compare_mode, 
                                    dirs_dict   = dirs_dict ,
                                    resize_fid   = args.resize_fid,
                                    calc_clipscore = args.calc_clipscore,
                                    vis_score_dist = True
                                    )       

    
    elif args.compare_vis:
        vis_metrics_for_methods((all_unmaps, all_latents, time_steps_sorted), 
                                args.methods_eval, 
                                compare_mode = args.compare_mode, 
                                dirs_dict   = dirs_dict ,
                                resize_fid   = args.resize_fid,
                                images_path = images_path,
                                backup_best_worst = True,
                                jump_to_vis = True,
                                )


    else:
        eval_metrics_for_methods((all_unmaps, all_latents, time_steps_sorted), 
                                    args.methods_eval, 
                                    compare_mode = args.compare_mode, 
                                    dirs_dict   = dirs_dict ,
                                    resize_fid   = args.resize_fid,
                                    calc_clipscore = args.calc_clipscore,
                                    calc_cmmd =  args.calc_cmmd,
                                    calc_sup_metrics = args.calc_sup_metrics,
                                    calc_grad_fid = args.calc_grad_fid
                                    )








def analyze_compare_methods(args):
    
    collect_and_merge_results(args.output_dir_compare)
    print_stats(args.output_dir_compare)
    exit(1)

    

def generate_uncertaintity_samples(args):
    deterministic(2024)
    

    dataset = args.loaded_dataset #load_dataset("jxie/flickr8k", split=f"validation[:{NUM_SAMPLES_TO_GENERATE}]", trust_remote_code=True) 
    
    
    flag_cant_resume = True

    start_iter = 0
    end_iter = len(dataset)#len(dataset)
    
    sample_idx = 0
    if args.partition:
        start_iter = args.partition
        end_iter = len(dataset)#start_iter + 10000
        sample_idx = args.partition 
    for batch_start in range(start_iter, end_iter, args.batch_size):
        if flag_cant_resume:
            
            if False:
                output_file_to_check = os.path.join(args.output_dir, str(batch_start+args.batch_size),"var_uc.json" )
                if os.path.isfile(output_file_to_check):
                    print(output_file_to_check)
                    sample_idx+= args.batch_size
                    continue 
                else:
                    flag_cant_resume = False
            else:
                output_dir_to_check = os.path.join(args.output_dir, str(batch_start))
                if os.path.isdir(output_dir_to_check):
                    sample_idx+= args.batch_size
                    continue 
                
                else:
                    print(f"found {output_dir_to_check}")
                    
           
                    
      
        batch_end = min(batch_start + args.batch_size, len(dataset))
        batch_items = dataset[batch_start:batch_end]
        
        # Select shortest caption for each item in batch
        prompts = []

        if args.dataset == "flickr8k":
            for i in range(len(batch_items['caption_0'])):
                # Get all captions for this item
                captions = [batch_items[f'caption_{j}'][i] for j in range(5) if f'caption_{j}' in batch_items]
                # Select shortest
                shortest_caption = min(captions, key=len)
                prompts.append(shortest_caption)
        elif args.dataset == "coco":
            prompts = batch_items['caption']
            
        # Generate images
        output = args.pipe(prompts, apply_uc = args.apply_uc, apply_uc_on_all_timesteps=args.apply_uc_on_all_timesteps, 
                            return_mid_reps = args.return_mid_reps, return_aux = True,
                            apply_var_method_uc= args.generate_var_uc_scores)

        
        
            
        
        images = output[0].images
        uncertainty_maps = output[1]["uncertainty_maps"]
        latents_lst = output[1]["latents_lst"]
        pixel_wise_uncertainty_lst = output[1]["pixel_wise_uncertainty_lst"]

        aux_list = output[1]["aux_list"]

        aux_list = torch.stack(aux_list,dim=0)
        aux_list = aux_list.permute(1,0,2,3)
        timesteps = sorted(uncertainty_maps.keys(), reverse=True)
        if args.model == "PixArt":
            timesteps = timesteps[9:17]
            aux_list = aux_list[:,9:17]
            model_path = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
        elif args.model == "SDXL":
            timesteps = timesteps[11:-5]
            aux_list = aux_list[:,12:-5]
            model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        elif args.model == "1.5v":
            timesteps = timesteps[11:-5]
            aux_list = aux_list[:,12:-5]
            model_path = "runwayml/stable-diffusion-v1-5"
        

        
        

        score_ablation, _ = generate_special_batch_map_globalTimestep(aux_list, 'prWeighted', model_path, timesteps)
        #print(score_ablation)
        #exit(1)
        if args.generate_var_uc_scores:
            var_uc_results = []
            for inner in pixel_wise_uncertainty_lst:
                stacked = torch.stack(inner, dim=1)  # [B,N,4,64,64]

               
                summed = stacked.sum(dim=1).sum(dim=1)  # -> [B,64,64]

                var_uc_results.append(summed.float())
            
            var_uc_results = (torch.stack(var_uc_results, dim=1)).sum(dim=(-1, -2))
           
        
        # Save each image with its prompt
        sample_idx_copy = sample_idx
        for idx in range(len(images)):
            # Create subdirectory for this sample
            sample_dir = os.path.join(args.output_dir, str(sample_idx))
            os.makedirs(sample_dir, exist_ok=True)


            with open(os.path.join(sample_dir, "ablation.json"), "w") as f:
                json.dump(float(score_ablation[idx].item()), f, indent=4)

            if args.generate_var_uc_scores:
                var_uc_results_sample = var_uc_results[idx]
                d_var_uc_results = {i: var_uc_results_sample[i].item() for i in range(len(var_uc_results_sample))}
                with open(os.path.join(sample_dir, "var_uc.json"), "w") as f:
                    json.dump(d_var_uc_results, f, indent=4)
                
                #sample_idx+=1
                #continue


            
            # Save prompt to txt file
            with open(os.path.join(sample_dir, "prompt.txt"), "w") as f:
                f.write(prompts[idx])
            
            # Save image
            images[idx].save(os.path.join(sample_dir, "output.jpg"), quality=95)
            
            print(f"Sample {sample_idx}: {prompts[idx]}")
            sample_idx += 1

        
        if False: #args.generate_var_uc_scores == False:
            save_uncertainty_maps(
                uncertainty_maps, 
                sample_idx_copy,
                latents_lst,
                out_dir = args.output_dir,
                cmap    = "hot",
                dpi=150,
            )
        #exit(1)
        #exit(1)
    


def eval_var_uc(args):
   
    generated_dataset_dir = args.output_dir
    real_dataset_dir = args.real_dataset_dir
    


    subdirs = sorted([d for d in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, d))], 
                    key=lambda x: int(x))
    
    if args.calc_sup_metrics:
        subdirs = subdirs[:10000]
    
    
    segments = range(12)

    if args.calc_grad_fid:
        segments = [9,11]
    for seg in segments:
        sample_score_mapper = {}
        method = f"VARUC_{seg}"
        
        final_output_dir = f"{args.output_dir_compare}/{method}"
        if  args.calc_grad_fid == False:
            if check_results_exists(args.output_dir_compare, method) == True:
                continue

       
        for idx, subdir in enumerate(subdirs):

            subdir_path = os.path.join(args.output_dir, subdir)
            
            with open(f"{subdir_path}/var_uc.json", "r") as f:
                data = json.load(f)
            
            sample_score_mapper[idx] = data[str(seg)]
            
        
        if args.calc_grad_fid:
            sorted_file_indices = sorted(sample_score_mapper, key=sample_score_mapper.get, reverse=False)
            len_file_ids = len(sorted_file_indices)
            d_res = {}
            
            for i in range(5):
                start = i * len_file_ids // 5
                end = (i + 1) * len_file_ids // 5
                partition = sorted_file_indices[:start] + sorted_file_indices[end:]
                
                d_res[i] = compute_fid_custom(generated_dataset_dir, real_dataset_dir, file_indices=partition)
            
            update_json(f"tmp/res_{method}.json", d_res)
            continue
        

        if args.calc_grad_fid:
            exit(1)
        
        sorted_file_indices = sorted(sample_score_mapper, key=sample_score_mapper.get, reverse=True)
        

        sorted_file_indices = sorted_file_indices[int(0.16 * len(sorted_file_indices)):]
        
        if args.calc_sup_metrics:
            d = {}
            pick_score, hpsv2_score, aes_score = run_sup_metrics(generated_dataset_dir, file_indices = sorted_file_indices)
            d["pick_score"] = pick_score
            d["hpsv2_score"] = hpsv2_score 
            d["aes_score"] = aes_score
            update_json(f"{final_output_dir}/res.json", d)
            continue
        
        d_res = {}
        fid_res = compute_fid_custom(generated_dataset_dir, real_dataset_dir, file_indices=sorted_file_indices)
        
        
        '''prec_rec_res = calculate_metrics(
            real_folder=real_dataset_dir,
            gen_folder=generated_dataset_dir,
            nhood_size=3,
            batch_size=32,
            file_indices = sorted_file_indices
        )

        d_res["precision"] = prec_rec_res["precision"]
        d_res["recall"] = prec_rec_res["recall"]'''
        d_res["fid"] = fid_res

        update_json(f"{final_output_dir}/res.json", d_res)
        
        



def eval_ablation(args):
   
    generated_dataset_dir = args.output_dir
    real_dataset_dir = args.real_dataset_dir
    

    subdirs = sorted([d for d in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, d))], 
                    key=lambda x: int(x))
    
    if args.calc_sup_metrics:
        subdirs = subdirs[:10000]
    

   
    sample_score_mapper = {}
    method = f"ablation"
        
    final_output_dir = f"{args.output_dir_compare}/{method}"
    if args.calc_grad_fid == False:
        if check_results_exists(args.output_dir_compare, method) == True:
            exit(1)

       
    for idx, subdir in enumerate(subdirs):

        subdir_path = os.path.join(args.output_dir, subdir)
        
        with open(f"{subdir_path}/ablation.json", "r") as f:
            data = json.load(f)
        
        
        sample_score_mapper[idx] = data
            
    sorted_file_indices = sorted(sample_score_mapper, key=sample_score_mapper.get, reverse=True)
    sorted_file_indices_sup = sorted_file_indices[int(0.16 * 10000) :10000]

    sorted_file_indices = sorted_file_indices[int(0.16 * len(sorted_file_indices)):]
        
    d = {}
    pick_score, hpsv2_score, aes_score = run_sup_metrics(generated_dataset_dir, file_indices = sorted_file_indices_sup)
    d["pick_score"] = pick_score
    d["hpsv2_score"] = hpsv2_score 
    d["aes_score"] = aes_score
    d["fid"] = compute_fid_custom(generated_dataset_dir, real_dataset_dir, file_indices=sorted_file_indices)

    update_json(f"{final_output_dir}/res.json", d)
    exit(1)
        
       




def collate_fn_heatmap_eval(batch):
    image_paths = [item['image_path'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    prompts = [item['prompt'] for item in batch]
    output_dirs = [item['output_dir'] for item in batch]
    return images, prompts, output_dirs, image_paths 

def generate_eval_heatmaps(args):
    import numpy as np
    model = RAHF()
    ckpt_path = 'artifacts_heatmap_generator/RichHF/rahf_model.pt'
    model.load_state_dict(torch.load(ckpt_path,map_location='cuda') )
    model.eval()
    
    # Create dataset and dataloader
    dataset = HeatmapEvalDataset(args.manual_prepare)
    dataloader = DataLoader(
        dataset, 
        batch_size=8,  # Adjust based on your GPU memory
        shuffle=False, 
        num_workers=4,  # Adjust based on your CPU cores
        collate_fn=collate_fn_heatmap_eval
    )
    
    with torch.no_grad():
        for images, prompts, output_dirs, image_paths in dataloader:
            # Forward pass on batch
            images = images.squeeze(1)
            
            out = model(images, prompts)
            heatmaps_batch = out.pop('heatmaps')
            
            # Save results for each item in batch
            for i, output_dir in enumerate(output_dirs):

                fig, axes = plt.subplots(1, 2, figsize=(6 * 2, 5))
                c = 0
                for k in heatmaps_batch:
                    # Extract i-th sample from batch
                    heatmap = heatmaps_batch[k][i]
                    torch.save(heatmap, f"{output_dir}/{k}.pt")

                    orig_img = Image.open(image_paths[i]).convert("RGB")
                    orig_np = np.array(orig_img)
                    heat = heatmap.detach().cpu().squeeze().numpy()
                    print(heat.shape)
                    heat = np.array(Image.fromarray(heat).resize((1024, 1024), Image.BILINEAR))

                    axes[c].imshow(orig_np)
                    axes[c].imshow(heat, cmap="hot", alpha=0.5)
                    axes[c].axis("off")
                    axes[c].set_title(k, fontsize=12)
                    c+=1

                plt.tight_layout()
                plt.savefig(f"{output_dir}/overlay.png", dpi=200)
                plt.close()
                print(output_dir)

            




def calc_iou(args):

    from sklearn.metrics import precision_recall_curve, auc

    op = 2
    precision = True
    iou_rahf = 0
    iou_mine = 0
    iou_ablation = 0

    iou_rahf_lst = []
    iou_mine_lst = []
    iou_ablation_lst = []

    all_gt = []
    all_method = []
    #all_method = []
    all_baseline = []

    d_boundaryF1 = {"mine": 0, "ab": 0, "rahf": 0}
    d_hausdorff = {"mine": 0, "ab": 0, "rahf": 0}
    d_auroc = {"mine": 0, "ab": 0, "rahf": 0}
    d_AP = {"mine": 0, "ab": 0, "rahf": 0}


    gt_path = "gt_segmentations/SegmentationClass"
    gt_path_check = "gt_segmentations/check"

    mine_path = f"uncertaintity_maps_demo/manual_prepare/{args.model}"
    overall_sum_iou = 0
    overall_sum_rahf = 0

    def calculate_iou(pred_binary, gt_binary):
      
        intersection = (pred_binary * gt_binary).sum()
        union = ((pred_binary + gt_binary) > 0).float().sum()
        iou = (intersection / (union + 1e-8)).item()
        if op == 1:
            if precision:
                intersection, gt_binary.sum().float().item()
            else:

                return intersection, union #union
        else:
            if precision:
                return (intersection / (gt_binary.sum().float().item() + 1e-8)).item(), _
            return iou, _
    
    count = 0
    #print(len(sorted(os.listdir(gt_path), key=lambda x: int(os.path.splitext(x)[0]))))
    #exit(1)
    gen_grpah = True
   
    gen_iou_graph = True

    for file_name in sorted(os.listdir(gt_path), key=lambda x: int(os.path.splitext(x)[0])):
       
        file_number = file_name.split(".")[0]
        
        origin_img = Image.open(f"{mine_path}/{file_number}/output.jpg")
        img = Image.open(os.path.join(gt_path, file_name)).convert("L")
        mask_gt = torch.from_numpy(np.array(img) > 0).int() # 2D seg map

        #print(mask_gt)
        #exit(1)

        subdir_path = f"{mine_path}/{file_number}"
        unmap_files = [f for f in os.listdir(subdir_path) if f.endswith("_unmap.pt")]
        unmap_files.sort(key=lambda x: int(x.split("_")[0]), reverse=True)
        time_steps_sorted = [int(elem.split("_")[0]) for elem in  unmap_files]
        
        unmaps = [torch.load(os.path.join(subdir_path, f) )for f in unmap_files]
        
        latent_files = [f for f in os.listdir(subdir_path) if f.endswith("_latent.pt")]
        latent_files.sort(key=lambda x: int(x.split("_")[0]), reverse=True)
        latents = [torch.load(os.path.join(subdir_path, f) )for f in latent_files]
        model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        latents = torch.stack(latents,dim=0).unsqueeze(0)
        unmaps = torch.stack(unmaps,dim=0).unsqueeze(0)

        #print(unmaps.shape)
        #print(latents.shape)


        time_steps_sorted = time_steps_sorted[11:-5]
        unmaps = unmaps[:,12:-5]
        latents = latents[:,12:-5]


        _, mask_mine_map = generate_special_batch_map_globalTimestep(unmaps, 'pr', model_path, time_steps_sorted)
        _, mask_mine_ablation = generate_special_batch_map_globalTimestep(latents, 'pr', model_path, time_steps_sorted)

        RAHF_map = torch.load(f"{subdir_path}/implausibility.pt")

  
        '''save_overlay(
                    image_pil=origin_img,
                    heatmap=mask_mine_map,
                    out_path=f"{gt_path_check}/output{f}_mine.jpg",
                    alpha=0.75,
                    cmap="hot",
                    target_size = 512 if args.model == "1.5v" else 1024
                )

        save_overlay(
                    image_pil=origin_img,
                    heatmap=mask_mine_ablation,
                    out_path=f"{gt_path_check}/output{f}_ablation.jpg",
                    alpha=0.75,
                    cmap="hot",
                    target_size = 512 if args.model == "1.5v" else 1024
                )        

        save_overlay(
                    image_pil=origin_img,
                    heatmap=RAHF_map,
                    out_path=f"{gt_path_check}/output{f}_RAHF.jpg",
                    alpha=0.75,
                    cmap="hot",
                    target_size = 512 if args.model == "1.5v" else 1024
                )'''


    

        target_size = 1024

        mask_mine_map = F.interpolate(
                        mask_mine_map.unsqueeze(0), 
                        size=target_size, 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0).squeeze(0).cpu()

        mask_mine_ablation =F.interpolate(
                        mask_mine_ablation.unsqueeze(0), 
                        size=target_size, 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0).squeeze(0).cpu()

        RAHF_map = F.interpolate(
                        RAHF_map.unsqueeze(0).unsqueeze(0), 
                        size=target_size, 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0).squeeze(0).cpu()

        
        
        count +=1


        mask_mine_map = (mask_mine_map - mask_mine_map.min()) / (mask_mine_map.max() - mask_mine_map.min())
        mask_mine_ablation = (mask_mine_ablation - mask_mine_ablation.min()) / (mask_mine_ablation.max() - mask_mine_ablation.min())
        RAHF_map = (RAHF_map - RAHF_map.min()) / (RAHF_map.max() - RAHF_map.min())
        
        if gen_grpah:
            all_gt.append(mask_gt.cpu().flatten())
            all_method.append(mask_mine_map.flatten())
            all_baseline.append(RAHF_map.flatten())
            continue

        #d_auroc["mine"] += auroc(mask_mine_map, mask_gt.cpu())
        #d_auroc["ab"]   += auroc(mask_mine_ablation, mask_gt.cpu())
        #d_auroc["rahf"] += auroc(RAHF_map, mask_gt.cpu())

        d_AP["mine"] += average_precision(mask_mine_map, mask_gt.cpu())
        d_AP["ab"]   += average_precision(mask_mine_ablation, mask_gt.cpu())
        d_AP["rahf"] += average_precision(RAHF_map, mask_gt.cpu())

        
        #mask_mine_map_binary = (mask_mine_map > mask_mine_map.mean()).float()
        #mask_mine_ablation_binary = (mask_mine_ablation > mask_mine_ablation.mean()).float()
        #RAHF_map_binary = (RAHF_map > RAHF_map.mean()).float()

        mask_mine_map_binary = (mask_mine_map > otsu_threshold(mask_mine_map.cpu().detach().numpy())).float()
        mask_mine_ablation_binary = (mask_mine_ablation > otsu_threshold(mask_mine_ablation.cpu().detach().numpy())).float()
        RAHF_map_binary = (RAHF_map > otsu_threshold(RAHF_map.cpu().detach().numpy())).float()

        #mask_mine_map_binary = (mask_mine_map > torch.quantile(mask_mine_map.to(torch.float32), 0.99)).float()
        #mask_mine_ablation_binary = (mask_mine_ablation > torch.quantile(mask_mine_ablation.to(torch.float32), 0.99)).float()
        #RAHF_map_binary = (RAHF_map > torch.quantile(RAHF_map.to(torch.float32), 0.99)).float()
        
        
        d_boundaryF1["mine"] += boundary_f1(mask_mine_map_binary, mask_gt.cpu(), dilation=2)
        d_boundaryF1["ab"]   += boundary_f1(mask_mine_ablation_binary, mask_gt.cpu(), dilation=2)
        d_boundaryF1["rahf"] += boundary_f1(RAHF_map_binary, mask_gt.cpu(), dilation=2)

       
        #d_hausdorff["mine"] += hausdorff_distance(mask_mine_map_binary, mask_gt.cpu())
        #d_hausdorff["ab"]   += hausdorff_distance(mask_mine_ablation_binary, mask_gt.cpu())
        #d_hausdorff["rahf"] += hausdorff_distance(RAHF_map_binary, mask_gt.cpu())



        if True:
            save_overlay_row(
                image_pil=origin_img,
                heatmaps=[
                    mask_gt,
                    RAHF_map_binary,
                    mask_mine_map_binary,
                    mask_mine_ablation_binary,
                    
                    
                ],
                out_path=f"{gt_path_check}/output{file_name}_comparison.jpg",
                alpha=0.75,
                cmap="hot",
                target_size=512 if args.model == "1.5v" else 1024,
                gap=0.03,
            )
        

        if op == 1:
            iou_mine_res = calculate_iou(mask_mine_map_binary, mask_gt.cpu())
            iou_mine     += iou_mine_res[0]
            overall_sum_iou  +=iou_mine_res[1]

            iou_ablation+= calculate_iou(mask_mine_ablation_binary, mask_gt.cpu())[0]
            iou_rahf_res = calculate_iou(RAHF_map_binary, mask_gt.cpu())
            iou_rahf    += iou_rahf_res[0]
            overall_sum_rahf  +=iou_rahf_res[1]
        else:
            iou_mine_lst.append(calculate_iou(mask_mine_map_binary, mask_gt.cpu())[0])
            iou_ablation_lst.append(calculate_iou(mask_mine_ablation_binary, mask_gt.cpu())[0])
            iou_rahf_lst.append(calculate_iou(RAHF_map_binary, mask_gt.cpu())[0])

    
    if gen_grpah:

        if gen_iou_graph:
            thresholds = np.linspace(0.5, 1, 100)

                        # Concatenate all samples and move to GPU
            all_gt = torch.cat(all_gt).cuda().bool()
            all_method = torch.cat(all_method).cuda()
            all_baseline = torch.cat(all_baseline).cuda()

            # Define thresholds to test
            thresholds = torch.linspace(0, 1, 1000).cuda()

            # Compute IoU for each threshold
            iou_method = []
            iou_baseline = []

            for thresh in thresholds:
                print(thresh)
                # Method
                pred_method = (all_method >= thresh)
                intersection_method = (pred_method & all_gt).sum()
                union_method = (pred_method | all_gt).sum()
                iou_method.append((intersection_method / union_method).item() if union_method > 0 else 0)
                
                # Baseline
                pred_baseline = (all_baseline >= thresh)
                intersection_baseline = (pred_baseline & all_gt).sum()
                union_baseline = (pred_baseline | all_gt).sum()
                iou_baseline.append((intersection_baseline / union_baseline).item() if union_baseline > 0 else 0)

            # Convert back to numpy for plotting
            thresholds_np = thresholds.cpu().numpy()

            # Plot: Threshold vs IoU
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds_np, iou_method, 'b-', linewidth=2, label='Method')
            plt.plot(thresholds_np, iou_baseline, 'r-', linewidth=2, label='Baseline')
            plt.xlabel('Threshold', fontsize=12)
            plt.ylabel('IoU', fontsize=12)
            plt.title('IoU vs Threshold', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.savefig('iou_vs_threshold.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("Figure saved as 'iou_vs_threshold.png'")
            print(f"Method IoU at threshold 0.97: {iou_method[np.argmin(np.abs(thresholds - 0.97))]:.4f}")
            print(f"Baseline IoU at threshold 0.97: {iou_baseline[np.argmin(np.abs(thresholds - 0.97))]:.4f}")
            exit(1)


        all_gt = np.concatenate(all_gt)
        all_method = np.concatenate(all_method)
        all_baseline = np.concatenate(all_baseline)
        # Compute precision-recall curves
        precision_method, recall_method, thresholds_method = precision_recall_curve(all_gt, all_method)
        precision_baseline, recall_baseline, thresholds_baseline = precision_recall_curve(all_gt, all_baseline)

        # Calculate AUC
        auc_method = 0 #auc(recall_method, precision_method)
        auc_baseline = 0 # auc(recall_baseline, precision_baseline)


        precision_method = precision_method[:-1]
        recall_method = recall_method[:-1]
        precision_baseline = precision_baseline[:-1]
        recall_baseline = recall_baseline[:-1]

        # Plot: Threshold vs Recall
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds_method, precision_method, 'b-', linewidth=2, label='Method')
        plt.plot(thresholds_baseline, precision_baseline, 'r-', linewidth=2, label='Baseline')
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision vs Threshold', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig('precision_vs_threshold.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Figure saved as 'recall_vs_threshold.png'")
        print(f"Method recall at threshold 0.97: {recall_method[np.argmin(np.abs(thresholds_method - 0.97))]:.4f}")
        print(f"Baseline recall at threshold 0.97: {recall_baseline[np.argmin(np.abs(thresholds_baseline - 0.97))]:.4f}")

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(recall_method, precision_method, 'b-', linewidth=2, label=f'Method (AUC={auc_method:.3f})')
        plt.plot(recall_baseline, precision_baseline, 'r-', linewidth=2, label=f'Baseline (AUC={auc_baseline:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve Comparison', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig('precision_recall_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Method AP: {auc_method:.4f}")
        print(f"Baseline AP: {auc_baseline:.4f}")
        print("Figure saved as 'precision_recall_comparison.png'")
        exit(1)

    if op == 1:
        print(iou_mine.item() / overall_sum_iou)
        #print(sum(iou_ablation) / len(iou_ablation))
        print(iou_rahf.item() / overall_sum_rahf)
    else:
        print(sum(iou_mine_lst) /len(iou_mine_lst) )
        print(sum(iou_rahf_lst) /len(iou_rahf_lst) )
        print(sum(iou_ablation_lst) /len(iou_ablation_lst) )

        print("---------------------------------\n\n")

        print("AP")
        print(d_AP["mine"] / 100)
        print(d_AP["ab"] / 100)
        print(d_AP["rahf"] / 100)
        print("\n")
        print("auroc")

        print(d_auroc["mine"] / 100)
        print(d_auroc["ab"] / 100)
        print(d_auroc["rahf"] / 100)
        print("\n")
        print("boundaryF1")
        print(d_boundaryF1["mine"] / 100)
        print(d_boundaryF1["ab"] / 100)
        print(d_boundaryF1["rahf"] / 100)
        print("\n")
        print("hausdorff")

        print(d_hausdorff["mine"] / 100)
        print(d_hausdorff["ab"] / 100)
        print(d_hausdorff["rahf"] / 100)        
        



if __name__ == "__main__":
    args          = parse_args()
    set_config(args,gen_samples = (args.mode == "generate_uncertaintity_samples"))
    if args.demo_correct:
        demo_correct(args)
    elif args.mode == "demo":
        demo(args)
    elif args.mode == "generate_uncertaintity_samples":
        generate_uncertaintity_samples(args)
        #generate_eval_heatmaps(args)
    elif args.mode == "generate_eval_heatmaps":
        generate_eval_heatmaps(args)
    elif args.mode == "compare_methods":
        if args.calc_grad_fid:
            compare_methods_tmp(args)
        else:
            compare_methods(args)
    elif args.mode == "analyze_compare_methods":
        analyze_compare_methods(args)
    elif args.mode == "eval_var_uc":
        eval_var_uc(args)

    elif args.mode == "eval_ablation":
        eval_ablation(args)
    elif args.mode == "qualitative":
        qualitative(args)
    elif args.mode == "playground":
        playground(args)
    elif args.mode == "motivation_exp_quant":
        motivation_exp_quant(args)
    elif args.mode == "playground2":
        playground2(args)
    elif args.mode == "motivation_exp":
        motivation_exp(args)
    elif args.mode == "user_study":
        user_study(args)
    elif args.mode == "manual_prepare":
        manual_prepare(args)
    elif args.mode == "calc_iou":
        calc_iou(args)

'''
# Convert PIL images to tensors for CLIPScore
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])



image_tensors = torch.stack([preprocess(img) for img in images]).to("cuda")

# Evaluate with CLIPScore
clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to("cuda")
scores = clip_metric(image_tensors, prompts)

print("CLIP scores:", scores)'''
































'''
# Access the actual model components for visualization
unet = pipe.unet  # This is where you can add hooks/visualizations
vae = pipe.vae
text_encoder = pipe.text_encoder

#print(pipe)
#exit(1)

# Generate images
prompts = ["a photo of a dog kicking a ninja", "a photo of a cat", "a landscape painting"]
images = pipe(prompts).images
images[0].save("output.jpg", quality=95)
images[1].save("output1.jpg", quality=95)
images[2].save("output2.jpg", quality=95)

# Evaluate with CLIP
image_tensors = torch.stack([preprocess(img) for img in images]).to("cuda")

clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to("cuda")
for idx, img in enumerate(image_tensors):
    #print(image_tensors.shape)
    #print(img.shape)
    #exit(1)
    scores = clip_metric(img, prompts[idx])
    print(f"CLIP scores: {scores}")'''