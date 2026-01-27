import cmmd.distance as distance
import cmmd.embedding as embedding
import cmmd.io_util as io_util
import numpy as np
import random

_BATCH_SIZE = 256
_MAX_COUNT = -1
_REF_EMBED_FILE = "tmp/ref_coco.npy"


def compute_cmmd(ref_dir, eval_dir, filtered, ref_embed_file=None, batch_size=128, max_count=-1):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    embedding_model = embedding.ClipEmbeddingModel()

    if ref_dir and ref_embed_file:
        raise ValueError("`ref_dir` and `ref_embed_file` both cannot be set at the same time.")
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
        
        print("REF EMBEDS loaded")
    else:
        ref_embs = io_util.compute_embeddings_for_dir(ref_dir, embedding_model, batch_size, max_count).astype(
            "float32"
        )

    
    
    eval_embs = io_util.compute_embeddings_for_dir(eval_dir, embedding_model, batch_size, max_count, filtered).astype("float32")

    
    val = distance.mmd(ref_embs, eval_embs)
    return val.numpy()


def mainFunc(ref_dir, eval_dir, filtered):
    res = compute_cmmd(None, eval_dir,filtered, _REF_EMBED_FILE, _BATCH_SIZE, _MAX_COUNT)
    print(
        "The CMMD value is: "
        f" {res:.3f}"
    )
    return res


if __name__ == "__main__":
    ref_dir = "datasets/coco/val2014"
    gen_folders = ["uncertaintity_maps/1.5v/basic/coco/", "uncertaintity_maps/SDXL/basic/coco/", ]
    indices = range(30000)
    n_remove = int(len(indices) * 0.16)         # how many to remove
    to_remove = set(random.sample(indices, n_remove))  # pick exactly 16% randomly
    filtered = [x for x in indices if x not in to_remove]
    for gen_f in gen_folders:
        mainFunc(ref_dir, gen_f,filtered)