import sys
import pandas as pd
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
from emotional_mi_pipeline import EmotionAnalysisPipeline

class LogitLensAnalysis:
    """
    A class to perform various Logit Lens and Causal Tracing analyses on a 
    language model, specifically tailored for emotion classification tasks.
    It relies on a model that supports caching intermediate activations (like TransformerLens).
    """

    def __init__(self, mi_pipeline: EmotionAnalysisPipeline):
        
        self.mi_pipeline = mi_pipeline
        self.model = mi_pipeline.model
        self.tokenizer = mi_pipeline.tokenizer
        self.device = mi_pipeline.device
        

    def analyze_mlp_logit_contributions(self, prompt_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyzes a DataFrame of prompts by calculating logit scores and ranks for each layer,
        based on the MLP output contribution only.
        
        Returns both the full, un-averaged metrics and the averaged metrics per layer group.
        """
        all_metrics = []
        
        if not hasattr(self.model, 'cfg') or not hasattr(self.model.cfg, 'n_layers'):
            print("Model configuration is missing or incomplete (n_layers). Cannot proceed.", file=sys.stderr)
            return pd.DataFrame(), pd.DataFrame()

        for _, row in prompt_df.iterrows():
        
            prompt = row['constrained prompt']
            true_emotion = row['emotion']
            predicted_emotion = row['predicted emotion']
        
            try:
                
                true_ids = self.mi_pipeline.get_token_ids(true_emotion)
                predicted_ids = self.mi_pipeline.get_token_ids(predicted_emotion)
                
                input_ids = self.mi_pipeline.tokenizer.encode(prompt, return_tensors='pt')
                # The index of the token immediately following the prompt (the first prediction token)
                final_token_idx = input_ids.shape[-1] - 1
            
                
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(prompt)

                for layer_idx in range(self.model.cfg.n_layers):
                
                    # output of the MLP for the last token
                    mlp_out_contribution = cache[("mlp_out", layer_idx)][0, final_token_idx, :]

                    # MLP contribution through the unembedding matrix to get logit contribution
                    mlp_logits = self.model.unembed(mlp_out_contribution)
                
                    # Sum of the logits for the true and predicted emotion tokens.
                    # Assuming get_token_ids returns a list of token IDs
                    true_logit_raw = mlp_logits[true_ids].sum().item()
                    predicted_logit_raw = mlp_logits[predicted_ids].sum().item()
                    logit_difference = predicted_logit_raw - true_logit_raw
                
                    # rank for the true and predicted tokens
                    true_rank = self.mi_pipeline.get_rank(mlp_logits, true_ids)
                    predicted_rank = self.mi_pipeline.get_rank(mlp_logits, predicted_ids)
                
                    all_metrics.append({
                        'layer': layer_idx,
                        'true_emotion': true_emotion,
                        'predicted_emotion': predicted_emotion,
                        'true_logit_raw': true_logit_raw,
                        'predicted_logit_raw': predicted_logit_raw,
                        'logit_difference': logit_difference,
                        'true_rank': true_rank,
                        'predicted_rank': predicted_rank
                    })
            except Exception as e:
                print(f"Skipping analysis for prompt: {row['constrained prompt']}\nError: {e}", file=sys.stderr)
                continue

        if not all_metrics:
            print("No metrics were generated. The prompt DataFrame might be empty or a tokenization error occurred.", file=sys.stderr)
            return pd.DataFrame(), pd.DataFrame()
        
        all_metrics_df = pd.DataFrame(all_metrics)
    
        # Calculate the average metrics
        grouping_cols = ['layer', 'true_emotion', 'predicted_emotion']

        average_metrics_df = all_metrics_df.groupby(grouping_cols).agg(
        
            true_logit_raw_mean=('true_logit_raw','mean'),
            predicted_logit_raw_mean=('predicted_logit_raw','mean'), 
            true_logit_std=('true_logit_raw','std'),
            predicted_logit_std=('predicted_logit_raw','std'),
            logit_difference_mean = ('logit_difference', 'mean'),
            logit_difference_std = ('logit_difference', 'std')
        ).reset_index()
        
        return all_metrics_df, average_metrics_df

    def analyze_logit_single_prompt_mlp(self, prompt: str, true_emotion: str, predicted_emotion: str) -> pd.DataFrame:
        """
        Analyzes a single prompt by calculating MLP logit scores and ranks for each layer.
        """
        all_metrics = []
        
        if not hasattr(self.model, 'cfg') or not hasattr(self.model.cfg, 'n_layers'):
            print("Model configuration is missing or incomplete (n_layers). Cannot proceed.", file=sys.stderr)
            return pd.DataFrame()

        try:
        
            true_ids = self.mi_pipeline.get_token_ids(true_emotion)
            predicted_ids = self.mi_pipeline.get_token_ids(predicted_emotion)
        
            input_ids = self.mi_pipeline.tokenizer.encode(prompt, return_tensors='pt')
            final_token_idx = input_ids.shape[-1] - 1
        
            
            with torch.no_grad():
                # The model must support run_with_cache to use this analysis
                _, cache = self.mi_pipeline.model.run_with_cache(prompt)

            for layer_idx in range(self.model.cfg.n_layers):
                # output of the MLP for the last token
                mlp_out_contribution = cache[("mlp_out", layer_idx)][0, final_token_idx, :]

                # MLP contribution through the unembedding matrix
                mlp_logits = self.model.unembed(mlp_out_contribution)
            
                # Sum of the logits for the true and predicted emotion tokens.
                true_logit_raw = mlp_logits[true_ids].sum().item()
                predicted_logit_raw = mlp_logits[predicted_ids].sum().item()
                logit_difference = predicted_logit_raw - true_logit_raw
            
                # rank for the true and predicted tokens
                true_rank = self.mi_pipeline.get_rank(mlp_logits, true_ids)
                predicted_rank = self.mi_pipeline.get_rank(mlp_logits, predicted_ids)
            
                all_metrics.append({
                    'layer': layer_idx,
                    'true_logit_raw': true_logit_raw,
                    'predicted_logit_raw': predicted_logit_raw,
                    'logit_difference': logit_difference,
                    'true_rank': true_rank,
                    'predicted_rank': predicted_rank
                })
            
        except Exception as e:
            print(f"Failed to analyze prompt: {prompt}\nError: {e}", file=sys.stderr)
            return pd.DataFrame()
        
        return pd.DataFrame(all_metrics)

    def analyze_logit_final_embed_contributions(self, prompt_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyzes a DataFrame of prompts by calculating logit scores, logit differences, and ranks
        for every layer using the **Logit Lens technique (full residual stream + unembed)**.

        This unified function returns both the detailed, un-averaged metrics (for Causal Tracing)
        and the averaged metrics (for Logit Lens plotting).
        """
        all_metrics = []
        n_layers = self.model.cfg.n_layers

        for _, row in prompt_df.iterrows():

            prompt_text = row['constrained prompt']
            true_emotion = row['emotion']
            predicted_emotion = row['predicted emotion']
        
            try:            
                # get_unique_token_id for single-token targets
                true_id = self.mi_pipeline.get_unique_token_id(true_emotion)
                predicted_id = self.mi_pipeline.get_unique_token_id(predicted_emotion)

                #to ensure IDs are integers before proceeding
                if not isinstance(true_id, int) or not isinstance(predicted_id, int):
                    raise ValueError("Token IDs are not single integers.")

                input_ids = self.mi_pipeline.tokenizer.encode(prompt_text, return_tensors='pt')
                final_token_idx = input_ids.shape[-1] - 1
                
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(prompt_text)

            
                for layer_idx in range(n_layers):
                # Apply final LayerNorm and Unembed to the residual stream after this layer
                    current_residual_stream = cache[("resid_post", layer_idx)]
                    # Slice to the final token position
                    final_token_residual = current_residual_stream[0, final_token_idx, :]
                    
                    layer_logits = self.model.ln_final(final_token_residual)
                    layer_logits_final_token = self.model.unembed(layer_logits)

                # Logit Scores (using the single, unique token ID)
                    true_logit_raw = layer_logits_final_token[true_id].item()
                    predicted_logit_raw = layer_logits_final_token[predicted_id].item()
                    logit_difference = predicted_logit_raw - true_logit_raw # y = predicted - true

                # Calculate Ranks
                    true_rank = self.mi_pipeline.get_rank(layer_logits_final_token, [true_id])
                    predicted_rank = self.mi_pipeline.get_rank(layer_logits_final_token, [predicted_id])

                    all_metrics.append({
                        'prompt_text': prompt_text,
                        'true_emotion': true_emotion,
                        'predicted_emotion': predicted_emotion,
                        'layer': layer_idx,
                        'true_logit_raw': true_logit_raw,
                        'predicted_logit_raw': predicted_logit_raw,
                        'logit_difference': logit_difference,
                        'true_rank': true_rank,
                        'predicted_rank': predicted_rank,
                        'true_token_id': true_id,
                        'predicted_token_id': predicted_id,
                    })

            except Exception as e:
                print(f"Skipping analysis for prompt: {prompt_text}\nError: {e}", file=sys.stderr)
                continue

        if not all_metrics:
            print("No metrics were generated. Check your prompt DataFrame and tokenization functions.", file=sys.stderr)
            return pd.DataFrame(), pd.DataFrame()

        all_metrics_df = pd.DataFrame(all_metrics)

        # average metrics
        grouping_cols = ['layer', 'true_emotion', 'predicted_emotion']

        average_metrics_df = all_metrics_df.groupby(grouping_cols).agg(
        
            true_logit_raw_mean = ('true_logit_raw', 'mean'),
            predicted_logit_raw_mean = ('predicted_logit_raw', 'mean'),
            logit_difference_mean = ('logit_difference', 'mean'),
            true_logit_std = ('true_logit_raw', 'std'),
            predicted_logit_std = ('predicted_logit_raw', 'std'),
            logit_difference_std = ('logit_difference', 'std')
        ).reset_index()

        return all_metrics_df, average_metrics_df

    def get_sequence_log_likelihood(self, prompt_text: str, answer_tokens: List[int]) -> float:
        
        
        """
        Calculates the total log-likelihood for a multi-token answer sequence P(T1, T2, ... | prompt_text).
        This function correctly sums the log-probabilities for each token in the sequence.
        
        Note: This is generally only useful for multi-token answers, not single-token emotion targets.
        """
        if not answer_tokens:
            return -float('inf')


        
        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors='pt')[0].to(self.model.cfg.device)
        prompt_len = prompt_ids.shape[0]

        
        full_input_ids = torch.cat([prompt_ids, torch.tensor(answer_tokens).to(self.model.cfg.device)])
        
        with torch.no_grad():
            # Get the logits for the entire sequence.
            logits, _ = self.model.run_with_cache(full_input_ids)
        
        total_log_likelihood = 0.0
        

        # The prediction for token answer_tokens[i] is made at position prompt_len + i
        for i, token_id in enumerate(answer_tokens):
            prediction_pos = prompt_len + i 
            prediction_logits = logits[0, prediction_pos - 1, :] # The prediction at position N is for the token N+1
            log_probs = F.log_softmax(prediction_logits, dim=-1)
            log_prob = log_probs[token_id].item()
            total_log_likelihood += log_prob

        return total_log_likelihood



    def analyze_logit_LL_prompts(self, prompt_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyzes a DataFrame of prompts by calculating the Log-Likelihood Ratio (LLR) 
        for the first emotion token (T1) across every layer using the Logit Lens technique.
        This provides a mathematically robust, layer-specific causal trace.
        """
        all_metrics = []
        
        
        try:
            n_layers = self.model.cfg.n_layers
            
        except Exception as e:
            
            print(f"Could not determine n_layers from model config. Error: {e}", file=sys.stderr)

            return pd.DataFrame(), pd.DataFrame()

        for index, row in prompt_df.iterrows():

            prompt_text = row.get('constrained prompt', None)
            true_emotion = row.get('emotion', None)
            predicted_emotion = row.get('predicted emotion', None)

            if prompt_text is None or true_emotion is None or predicted_emotion is None:
                    
                    print(f"Skipping row {index}: Missing required columns ('constrained prompt', 'emotion', or 'predicted emotion').", file=sys.stderr)
                    
                    continue
            
            prompt_id = index 

            try:
                # Assuming get_token_ids returns [BOS_ID, TOKEN1_ID, ...]
                true_ids = self.mi_pipeline.get_token_ids(true_emotion)
                predicted_ids = self.mi_pipeline.get_token_ids(predicted_emotion)
            
                
                if not true_ids or not predicted_ids:
                    
                    raise ValueError(f"Emotion tokenization failed: True IDs={true_ids}, Predicted IDs={predicted_ids}. Ensure input text is correct.")

                
                true_id_t1 = true_ids[1] # Use the first actual token 
                predicted_id_t1 = predicted_ids[1]

                input_ids = self.mi_pipeline.tokenizer.encode(prompt_text, return_tensors='pt')
                final_token_idx = input_ids.shape[-1] - 1

            
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(prompt_text)

                # Logit Lens Analysis
                for layer_idx in range(n_layers):
                
                    current_residual_stream = cache[("resid_post", layer_idx)]
                    # Slice to the final token position
                    final_token_residual = current_residual_stream[0, final_token_idx, :]

                    layer_logits = self.model.ln_final(final_token_residual)
                    layer_logits_final_token = self.model.unembed(layer_logits)

                    # Raw Logits to Log-Probabilities 
                    log_probs = F.log_softmax(layer_logits_final_token, dim=-1) 

                    # Log-Likelihood (LL) for T1
                    true_LL_t1 = log_probs[true_id_t1].item()
                    predicted_LL_t1 = log_probs[predicted_id_t1].item()
                
                    # Log-Likelihood Ratio: y = LL_predicted - LL_true
                    log_likelihood_ratio = predicted_LL_t1 - true_LL_t1

                    all_metrics.append({
                        'prompt_id': prompt_id,
                        'prompt_text': prompt_text,
                        'true_emotion': true_emotion,
                        'predicted_emotion': predicted_emotion,
                        'layer': layer_idx,
                        'log_likelihood_ratio': log_likelihood_ratio,
                        'true_token_id': true_id_t1,
                        'predicted_token_id': predicted_id_t1,
                    })

            except Exception as e:
            
                print(f"Skipping analysis for prompt (ID {index}): {prompt_text}\nSpecific Error: {e}", file=sys.stderr)
                continue

        if not all_metrics:
            print("No metrics were generated. Check your prompt DataFrame and tokenization functions.", file=sys.stderr)
        
            return pd.DataFrame(), pd.DataFrame()

        all_metrics_df = pd.DataFrame(all_metrics)

    
        grouping_cols = ['layer', 'true_emotion', 'predicted_emotion']
    
        average_metrics_df = all_metrics_df.groupby(grouping_cols).agg(
            log_likelihood_ratio_mean=('log_likelihood_ratio', 'mean'),
            log_likelihood_ratio_std=('log_likelihood_ratio', 'std'),
        ).reset_index()
    
        average_metrics_df.rename(
            columns={'log_likelihood_ratio_mean': 'log_likelihood_ratio'}, 
            inplace=True
        )
    
        return all_metrics_df, average_metrics_df


    def analyze_logit_attention_distinction(self, prompt_df: pd.DataFrame, distractor_count: int = 100) -> pd.DataFrame:
        """
        Computes the relative attention-extracted attribute information, I_a^(l)(o), 
        from the paper "A Causal Lens for Interpretability."
        """
        all_metrics = []
        
        # Check if model has a config
        if not hasattr(self.model, 'cfg') or not hasattr(self.model.cfg, 'n_layers'):
            print("Model configuration is missing or incomplete (n_layers). Cannot proceed.", file=sys.stderr)
            return pd.DataFrame()

        # The unembedding matrix is W_U (or W_out from the final layer, depending on the library)
        unembedding_matrix = self.model.unembed.W_U.squeeze() 
        
        for _, row in prompt_df.iterrows():
            try:
                true_emotion_text = row['emotion']
                prompt_text = row['constrained prompt']

                true_ids = self.mi_pipeline.get_token_ids(true_emotion_text) 
                
                input_ids = self.tokenizer.encode(prompt_text, return_tensors='pt')
                final_token_idx = input_ids.shape[-1] - 1
                
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(prompt_text)

                for layer_idx in range(self.model.cfg.n_layers):
                    # MLP Logit Contribution to find top distractors
                    mlp_out_contribution = cache[("mlp_out", layer_idx)][0, final_token_idx, :]
                    mlp_logits = self.model.unembed(mlp_out_contribution)
                    
                    # top distractor token IDs (excluding the true token, though topk handles that by rank)
                    _, top_distractor_ids = torch.topk(mlp_logits, k=distractor_count)
                    
                    # Calculate the distinction vector V_A
                    true_unembedding_vector = unembedding_matrix[true_ids].mean(dim=0)
                    # V_distractor: mean unembedding vector of the top distractor tokens
                    distractor_unembedding_vectors = unembedding_matrix[top_distractor_ids]
                    mean_distractor_vector = distractor_unembedding_vectors.mean(dim=0)
                    
                    # Distinction vector (V_A = V_true - V_distractor)
                    distinction_vector = true_unembedding_vector - mean_distractor_vector
                    
                    #Compute the final score (Dot product with Attn-out)
                    attn_out_contribution = cache[("attn_out", layer_idx)][0, final_token_idx, :]
                    distinction_score = torch.dot(attn_out_contribution, distinction_vector).item()
                    
                    all_metrics.append({
                        'layer': layer_idx,
                        'distinction_score': distinction_score,
                    })
            except Exception as e:
                print(f"Skipping analysis for prompt: {row.get('constrained prompt', 'N/A')}\nError: {e}", file=sys.stderr)
                continue

        if not all_metrics:
            print("No metrics were generated. Check your prompt DataFrame and tokenization functions.", file=sys.stderr)
            return pd.DataFrame()
        
        all_metrics_df = pd.DataFrame(all_metrics)
        # Average the distinction score across all prompts for each layer
        average_metrics_df = all_metrics_df.groupby('layer').mean().reset_index()
        return average_metrics_df
