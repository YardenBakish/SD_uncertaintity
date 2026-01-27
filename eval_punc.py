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
             