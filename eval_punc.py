"""
PUNC (Prompt-based UNCertainty Estimation) Evaluation Implementation
Based on "Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation"

This implementation provides:
- Text-to-Image generation using Stable Diffusion 1.5
- Image captioning using BLIP-2
- ROUGE-L and ROUGE-1 (precision/recall) scoring
- BERTScore (precision/recall) scoring
- Uncertainty metrics: AUROC, AUPR, FPR95
"""

import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
import json
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

class PUNCEvaluator:
    """
    PUNC Evaluator for Text-to-Image Uncertainty Quantification
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize models for T2I generation and captioning"""
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load Stable Diffusion 1.5
        print("Loading Stable Diffusion 1.5...")
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            safety_checker=None
        ).to(device)
        
     
        
        self.caption_processor = AutoProcessor.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        self.caption_model = AutoModelForCausalLM.from_pretrained(
                'allenai/Molmo-7B-D-0924',
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
            ).to(device)
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rougeL'], 
            use_stemmer=True
        )
        
        print("All models loaded successfully!")


    
    def generate_image(self, prompt: str, num_inference_steps: int = 50, seed: int = None) -> torch.Tensor:
        """Generate image from text prompt"""
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        with torch.no_grad():
            image = self.sd_pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images[0]
        
        return image




    
    def caption_image(self, image_paths) -> str:
        inputs = self.caption_processor.process(
        images=[Image.open(image_path) for image_path in image_paths],
        text="Describe this image."
    )

        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=self.caption_processor.tokenizer
            )
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # print the generated text
        print(generated_text)
        exit(1)
        return generated_text


    def compute_rouge_scores(self, prompt: str, caption: str) -> Dict[str, float]:
        """
        Compute ROUGE-1 and ROUGE-L scores (precision and recall)
        
        Returns:
            Dictionary with rouge1_precision, rouge1_recall, rougeL_precision, rougeL_recall
        """
        scores = self.rouge_scorer.score(prompt, caption)
        
        return {
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall
        }
    
    
    def compute_bert_scores(self, prompts: List[str], captions: List[str]) -> Dict[str, np.ndarray]:
        """
        Compute BERTScore (precision and recall) for batch of prompt-caption pairs
        
        Returns:
            Dictionary with bert_precision and bert_recall arrays
        """
        P, R, F1 = bert_score(
            captions, 
            prompts, 
            lang='en',
            model_type='bert-base-uncased',
            device=self.device,
            verbose=False
        )
        
        return {
            'bert_precision': P.cpu().numpy(),
            'bert_recall': R.cpu().numpy()
        }
    

    def compute_uncertainty_score(self, prompt: str, caption: str, 
                                   metric_type: str = 'rougeL_recall') -> float:
        """
        Compute uncertainty score for a single prompt-caption pair
        
        Uncertainty = 1 - similarity_score
        
        Args:
            metric_type: One of ['rouge1_recall', 'rouge1_precision', 
                                 'rougeL_recall', 'rougeL_precision']
        """
        rouge_scores = self.compute_rouge_scores(prompt, caption)
        similarity = rouge_scores[metric_type]
        uncertainty = 1 - similarity
        return uncertainty
    

    def evaluate_dataset(self, prompts: List[str], is_ood: bool = True, 
                        num_inference_steps: int = 50, 
                        seed_start: int = 42) -> Dict[str, List[float]]:
        """
        Evaluate a dataset of prompts and return uncertainty scores
        
        Args:
            prompts: List of text prompts
            is_ood: Whether this is OOD dataset (affects which metrics to use)
            num_inference_steps: Number of diffusion steps
            seed_start: Starting seed for reproducibility
        
        Returns:
            Dictionary containing all uncertainty scores
        """
        results = {
            'rouge1_recall_unc': [],
            'rouge1_precision_unc': [],
            'rougeL_recall_unc': [],
            'rougeL_precision_unc': [],
            'captions': [],
            'prompts': []
        }
        
        print(f"\nEvaluating {len(prompts)} prompts...")
        for idx, prompt in enumerate(tqdm(prompts)):
            # Generate image
            image = self.generate_image(
                prompt, 
                num_inference_steps=num_inference_steps,
                seed=seed_start + idx
            )
            
            # Caption image
            caption = self.caption_image(image)
            
            # Compute ROUGE scores
            rouge_scores = self.compute_rouge_scores(prompt, caption)
            
            # Store uncertainties (1 - similarity)
            results['rouge1_recall_unc'].append(1 - rouge_scores['rouge1_recall'])
            results['rouge1_precision_unc'].append(1 - rouge_scores['rouge1_precision'])
            results['rougeL_recall_unc'].append(1 - rouge_scores['rougeL_recall'])
            results['rougeL_precision_unc'].append(1 - rouge_scores['rougeL_precision'])
            results['captions'].append(caption)
            results['prompts'].append(prompt)
        
        # Compute BERTScore in batch for efficiency
        print("Computing BERTScores...")
        bert_scores = self.compute_bert_scores(prompts, results['captions'])
        results['bert_recall_unc'] = (1 - bert_scores['bert_recall']).tolist()
        results['bert_precision_unc'] = (1 - bert_scores['bert_precision']).tolist()
        
        return results
    


    def compute_metrics(self, id_uncertainties: np.ndarray, 
                       ood_uncertainties: np.ndarray) -> Dict[str, float]:
        """
        Compute AUROC, AUPR, and FPR95 metrics
        
        Args:
            id_uncertainties: Uncertainty scores for in-distribution data
            ood_uncertainties: Uncertainty scores for out-of-distribution data
        
        Returns:
            Dictionary with auroc, aupr, fpr95
        """
        # Combine uncertainties and create labels
        # OOD should have higher uncertainty (positive class)
        uncertainties = np.concatenate([id_uncertainties, ood_uncertainties])
        labels = np.concatenate([
            np.zeros(len(id_uncertainties)),  # ID = 0
            np.ones(len(ood_uncertainties))   # OOD = 1
        ])
        
        # AUROC
        auroc = roc_auc_score(labels, uncertainties)
        
        # AUPR
        aupr = average_precision_score(labels, uncertainties)
        
        # FPR95 (False Positive Rate at 95% True Positive Rate)
        fpr, tpr, thresholds = roc_curve(labels, uncertainties)
        fpr95_idx = np.argmax(tpr >= 0.95)
        fpr95 = fpr[fpr95_idx]
        
        return {
            'auroc': auroc * 100,  # Convert to percentage
            'aupr': aupr * 100,
            'fpr95': fpr95 * 100
        }
    
    
    def full_evaluation(self, id_prompts: List[str], ood_prompts: List[str],
                       num_inference_steps: int = 50) -> Dict:
        """
        Perform full PUNC evaluation comparing ID and OOD datasets
        
        Returns:
            Dictionary with results for all metrics
        """
        print("="*80)
        print("PUNC EVALUATION - Stable Diffusion 1.5")
        print("="*80)
        
        # Evaluate ID dataset
        print("\n[1/2] Evaluating In-Distribution (Normal) Dataset...")
        id_results = self.evaluate_dataset(
            id_prompts, 
            is_ood=False,
            num_inference_steps=num_inference_steps
        )
        
        # Evaluate OOD dataset
        print("\n[2/2] Evaluating Out-of-Distribution Dataset...")
        ood_results = self.evaluate_dataset(
            ood_prompts,
            is_ood=True,
            num_inference_steps=num_inference_steps
        )
        
        # Compute metrics for each uncertainty type
        print("\n" + "="*80)
        print("COMPUTING METRICS")
        print("="*80)
        
        metrics_results = {}
        
        # For OOD detection, we use recall-based metrics (epistemic uncertainty)
        metric_names = [
            ('ROUGE-1 Recall', 'rouge1_recall_unc'),
            ('ROUGE-L Recall', 'rougeL_recall_unc'),
            ('BERT Recall', 'bert_recall_unc')
        ]
        
        for display_name, key in metric_names:
            metrics = self.compute_metrics(
                np.array(id_results[key]),
                np.array(ood_results[key])
            )
            metrics_results[display_name] = metrics
            
            print(f"\n{display_name}:")
            print(f"  AUROC: {metrics['auroc']:.2f}%")
            print(f"  AUPR:  {metrics['aupr']:.2f}%")
            print(f"  FPR95: {metrics['fpr95']:.2f}%")
        
        return {
            'id_results': id_results,
            'ood_results': ood_results,
            'metrics': metrics_results
        }


def create_sample_datasets():
    """
    Create sample datasets for demonstration
    You should replace these with actual datasets from the paper
    """
    
    # Normal (ID) - ImageNet-like prompts
    normal_prompts = [
        "A photograph of a golden retriever sitting in a grassy field",
        "A close-up photo of a red apple on a wooden table",
        "A picture of a blue sports car on a city street",
        "An image of a tabby cat sleeping on a couch",
        "A photo of a modern laptop computer on a desk"
    ]
    
    # Microscopic (OOD) - Simulated microscopic prompts
    microscopic_prompts = [
        "Microscopic view of red blood cells flowing through a capillary",
        "High magnification image of bacterial cells with flagella",
        "Electron microscope image of coronavirus particles",
        "Microscopic photograph of plant cell structures showing chloroplasts",
        "Detailed view of crystalline structures under polarized light microscopy"
    ]
    
    # Vague (Aleatoric) - Minimal context prompts
    vague_prompts = [
        "An image of a dog",
        "A picture of food",
        "An image of nature",
        "A photo of transportation",
        "An image of furniture"
    ]
    
    return {
        'normal': normal_prompts,
        'microscopic': microscopic_prompts,
        'vague': vague_prompts
    }


# ============================================================================
# MAIN EVALUATION SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Initialize evaluator
    evaluator = PUNCEvaluator()
    
    # Create sample datasets
    datasets = create_sample_datasets()
    
    # ========================================================================
    # TASK 1: OOD Detection (Normal vs Microscopic)
    # ========================================================================
    print("\n" + "="*80)
    print("TASK 1: OOD DETECTION - Normal (ID) vs Microscopic (OOD)")
    print("="*80)
    
    results_ood = evaluator.full_evaluation(
        id_prompts=datasets['normal'],
        ood_prompts=datasets['microscopic'],
        num_inference_steps=50
    )
    
    # ========================================================================
    # TASK 2: Aleatoric Uncertainty Detection (Normal vs Vague)
    # ========================================================================
    print("\n\n" + "="*80)
    print("TASK 2: ALEATORIC UNCERTAINTY - Normal vs Vague")
    print("="*80)
    
    # For aleatoric uncertainty, evaluate with precision-based metrics
    print("\n[1/2] Evaluating In-Distribution (Normal) Dataset...")
    id_results_vague = evaluator.evaluate_dataset(
        datasets['normal'],
        is_ood=False,
        num_inference_steps=50
    )
    
    print("\n[2/2] Evaluating Vague Prompts Dataset...")
    vague_results = evaluator.evaluate_dataset(
        datasets['vague'],
        is_ood=True,
        num_inference_steps=50
    )
    
    # Compute metrics using precision (for aleatoric uncertainty)
    print("\n" + "="*80)
    print("COMPUTING METRICS (Precision-based for Aleatoric Uncertainty)")
    print("="*80)
    
    metric_names_prec = [
        ('ROUGE-1 Precision', 'rouge1_precision_unc'),
        ('ROUGE-L Precision', 'rougeL_precision_unc'),
        ('BERT Precision', 'bert_precision_unc')
    ]
    
    for display_name, key in metric_names_prec:
        metrics = evaluator.compute_metrics(
            np.array(id_results_vague[key]),
            np.array(vague_results[key])
        )
        
        print(f"\n{display_name}:")
        print(f"  AUROC: {metrics['auroc']:.2f}%")
        print(f"  AUPR:  {metrics['aupr']:.2f}%")
        print(f"  FPR95: {metrics['fpr95']:.2f}%")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    
    # Save detailed results to JSON
    save_results = {
        'ood_task': {
            'id_prompts': datasets['normal'],
            'ood_prompts': datasets['microscopic'],
            'id_captions': results_ood['id_results']['captions'],
            'ood_captions': results_ood['ood_results']['captions'],
            'metrics': results_ood['metrics']
        },
        'vague_task': {
            'id_prompts': datasets['normal'],
            'vague_prompts': datasets['vague'],
            'id_captions': id_results_vague['captions'],
            'vague_captions': vague_results['captions'],
            'id_rouge1_prec_unc': id_results_vague['rouge1_precision_unc'],
            'vague_rouge1_prec_unc': vague_results['rouge1_precision_unc']
        }
    }
    
    with open('punc_evaluation_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print("\nResults saved to 'punc_evaluation_results.json'")
    print("\nTo use with larger datasets, replace the sample prompts in")
    print("create_sample_datasets() with your actual dataset prompts.")