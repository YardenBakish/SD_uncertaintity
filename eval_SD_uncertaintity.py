import torch
#from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from modules.pipeline_stable_diffusion import StableDiffusionPipeline
from torchmetrics.multimodal import CLIPScore
from PIL import Image
from datasets import load_dataset
from modules.unet_2D_conditioned import UNet2DConditionModel
from modules.scheduling_pndm import PNDMScheduler
from torchvision import transforms
import os
import torch.nn.functional as F
from eval_utils import *
from utils import *
from agg_experiments import get_artifact_mask
import matplotlib.pyplot as plt
from config import set_config

from metrics_clipscore import calculate_clipscore_metric
from metrics import compute_fid_custom
from metrics2 import calculate_metrics

from artifacts_heatmap_generator.RichHF.model import  preprocess_image, RAHF
import argparse

NUM_SAMPLES_TO_GENERATE = 1000

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
                                                                         'analyze_compare_methods',
                                                                         'eval_var_uc',
                                                                         'qualitative',
                                                                         'playground'])
    parser.add_argument('--model', type=str, default = '1.5v', choices = ['1.5v', 'SDXL', 'PixArt'])


    parser.add_argument('--resize_fid', type=int, default = 299, choices = [299, 512, 1024])
    parser.add_argument('--compare_mode', type=str, default = "fid_filter_high", choices = ["fid_filter_high"])
    parser.add_argument('--compare_vis', action='store_true')
    parser.add_argument('--calc_clipscore', action='store_true')
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


    args.output_vis_dir_compare = f"visualizations/compare/{args.dataset}/{args.model}/"





    if args.dataset == "coco":
        args.real_dataset_dir = "datasets/coco/val2014"

    args.batch_size = 16
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_demo, exist_ok=True)
    os.makedirs(args.output_dir_compare, exist_ok=True)
    os.makedirs(args.output_vis_dir_compare, exist_ok=True)
    os.makedirs(args.qualitative, exist_ok=True)




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
    apply_var_method_uc = False
   
    for start_idx in range(0, 16, args.batch_size):
        dataset = load_dataset("jxie/flickr8k", split=f"validation[{start_idx}:{start_idx+args.batch_size}]", trust_remote_code=True)  # take 5 examples for demo
        
        
        prompts = [item["caption_0"] for item in dataset]


        output = args.pipe(prompts)
        images = output[0]
        for idx in range(len(images)):
            print(prompts[idx])
            images[idx].save(f"{args.output_dir_demo}/output{start_idx+idx}.jpg", quality=95)
    
        
        exit(1)
        output = args.pipe(prompts, apply_uc = True, apply_uc_on_all_timesteps=True, return_mid_reps = True, apply_var_method_uc=apply_var_method_uc)

        images = output[0].images
        uncertainty_maps = output[1]["uncertainty_maps"]
        latents_lst = output[1]["latents_lst"]
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
            



def qualitative(args):
    deterministic(2024)
    apply_var_method_uc = False

    modelRAHF = RAHF()
    ckpt_path = 'artifacts_heatmap_generator/RichHF/rahf_model.pt'
    modelRAHF.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    modelRAHF.eval()
   
    for start_idx in range(0, 16, args.batch_size):
        dataset = load_dataset("jxie/flickr8k", split=f"validation[{start_idx}:{start_idx+args.batch_size}]", trust_remote_code=True)  # take 5 examples for demo
        
        
        prompts = [item["caption_0"] for item in dataset]
    
        output = args.pipe(prompts, apply_uc = True, apply_uc_on_all_timesteps=True, return_mid_reps = True, apply_var_method_uc=apply_var_method_uc)

        images = output[0].images

        

        
        uncertainty_maps = output[1]["uncertainty_maps"]
        latents_lst = output[1]["latents_lst"]
        if apply_var_method_uc:
            pixel_wise_uncertainty_lst = output[1]["pixel_wise_uncertainty_lst"][11]

        
            stacked_pixel_wise_uncertainty_lst = torch.stack(pixel_wise_uncertainty_lst, dim=1).sum(dim=1).sum(dim=1) 
       
        saved_image_paths = []
        for idx in range(len(images)):
            print(prompts[idx])
            images[idx].save(f"{args.qualitative}/output{start_idx+idx}.jpg", quality=95)
            saved_image_paths.append(f"{args.qualitative}/output{start_idx+idx}.jpg")
            if apply_var_method_uc:
                #summed = sum(pixel_wise_uncertainty_lst)
                #print(summed.shape)
                
                heatmap = torch.abs(stacked_pixel_wise_uncertainty_lst[idx]) #summed.sum(dim=1)[0]

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

        """image_rahf = torch.stack([preprocess_image(im) for im in saved_image_paths])
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
            )"""
        
        timesteps = sorted(uncertainty_maps.keys(), reverse=True)

    
        n_timesteps = len(timesteps)
        example_ts = timesteps[0]

        # Number of samples
        n_cols = len(uncertainty_maps[example_ts])#uncertainty_maps[example_ts][0].shape[0] // 2
        n_cols = 1 + n_cols
        last_layer_idx = n_cols - 2  # Last uncertainty map column
        num_samples = len(images)
        masks = prepare_culumative(num_samples, last_layer_idx, uncertainty_maps, timesteps)

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
        ASCED_masks =   get_artifact_mask(
                            torch.stack(latents_lst[10:24], dim=1),
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
        exit(1)     
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
                                    calc_clipscore = args.calc_clipscore
                                    )
    




def analyze_compare_methods(args):
    
    #collect_and_merge_results(args.output_dir_compare)
    print_stats(args.output_dir_compare)
    exit(1)

    

def generate_uncertaintity_samples(args):
    deterministic(2024)
    

    dataset = args.loaded_dataset #load_dataset("jxie/flickr8k", split=f"validation[:{NUM_SAMPLES_TO_GENERATE}]", trust_remote_code=True) 
    
    sample_idx = 0
    flag_cant_resume = True
    for batch_start in range(0, len(dataset), args.batch_size):
        if flag_cant_resume:
            
            if args.generate_var_uc_scores:
                output_file_to_check = os.path.join(args.output_dir, str(batch_start+args.batch_size),"var_uc.json" )
                if os.path.isfile(output_file_to_check):
                    print(output_file_to_check)
                    sample_idx+= args.batch_size
                    continue 
                else:
                    flag_cant_resume = False
            else:
                output_dir_to_check = os.path.join(args.output_dir, str(batch_start+args.batch_size))
                if os.path.isdir(output_dir_to_check):
                    sample_idx+= args.batch_size
                    continue 
                else:
                    flag_cant_resume = False
        
        
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
                            return_mid_reps = args.return_mid_reps,
                            apply_var_method_uc= args.generate_var_uc_scores)

        
        
            
        
        images = output[0].images
        uncertainty_maps = output[1]["uncertainty_maps"]
        latents_lst = output[1]["latents_lst"]
        pixel_wise_uncertainty_lst = output[1]["pixel_wise_uncertainty_lst"]

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


            if args.generate_var_uc_scores:
                var_uc_results_sample = var_uc_results[idx]
                d_var_uc_results = {i: var_uc_results_sample[i].item() for i in range(len(var_uc_results_sample))}
                with open(os.path.join(sample_dir, "var_uc.json"), "w") as f:
                    json.dump(d_var_uc_results, f, indent=4)
                
                sample_idx+=1
                continue


            
            # Save prompt to txt file
            with open(os.path.join(sample_dir, "prompt.txt"), "w") as f:
                f.write(prompts[idx])
            
            # Save image
            images[idx].save(os.path.join(sample_dir, "output.jpg"), quality=95)
            
            print(f"Sample {sample_idx}: {prompts[idx]}")
            sample_idx += 1

        
        if args.generate_var_uc_scores == False:
            save_uncertainty_maps(
                uncertainty_maps, 
                sample_idx_copy,
                latents_lst,
                out_dir = args.output_dir,
                cmap    = "hot",
                dpi=150,
            )
        #exit(1)
    


def eval_var_uc(args):
    generated_dataset_dir = args.output_dir
    real_dataset_dir = args.real_dataset_dir
    


    subdirs = sorted([d for d in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, d))], 
                    key=lambda x: int(x))
    
    #subdirs = subdirs[:100]
    
    
    segments = range(12)
    for seg in segments:
        sample_score_mapper = {}
        method = f"VARUC_{seg}"
        
        final_output_dir = f"{args.output_dir_compare}/{method}"
        if check_results_exists(args.output_dir_compare, method) == True:
            continue

        print(final_output_dir)
        for idx, subdir in enumerate(subdirs):

            subdir_path = os.path.join(args.output_dir, subdir)
            
            with open(f"{subdir_path}/var_uc.json", "r") as f:
                data = json.load(f)
            
            sample_score_mapper[idx] = data[str(seg)]
            
        sorted_file_indices = sorted(sample_score_mapper, key=sample_score_mapper.get, reverse=True)

        sorted_file_indices = sorted_file_indices[int(0.16 * len(sorted_file_indices)):]
        
        
        d_res = {}
        fid_res = compute_fid_custom(generated_dataset_dir, real_dataset_dir, file_indices=sorted_file_indices)
        prec_rec_res = calculate_metrics(
            real_folder=real_dataset_dir,
            gen_folder=generated_dataset_dir,
            nhood_size=3,
            batch_size=32,
            file_indices = sorted_file_indices
        )

        d_res["precision"] = prec_rec_res["precision"]
        d_res["recall"] = prec_rec_res["recall"]
        d_res["fid"] = fid_res

        update_json(f"{final_output_dir}/res.json", d_res)
        
        




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
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    
    # Create dataset and dataloader
    dataset = HeatmapEvalDataset(args.output_dir)
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
                    heat = np.array(Image.fromarray(heat).resize((512, 512), Image.BILINEAR))

                    axes[c].imshow(orig_np)
                    axes[c].imshow(heat, cmap="hot", alpha=0.5)
                    axes[c].axis("off")
                    axes[c].set_title(k, fontsize=12)
                    c+=1

                plt.tight_layout()
                plt.savefig(f"{output_dir}/overlay.png", dpi=200)
                plt.close()
                print(output_dir)

            
            
           

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
        compare_methods(args)
    elif args.mode == "analyze_compare_methods":
        analyze_compare_methods(args)
    elif args.mode == "eval_var_uc":
        eval_var_uc(args)
    elif args.mode == "qualitative":
        qualitative(args)
    elif args.mode == "playground":
        playground(args)


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