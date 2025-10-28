import pandas as pd
import numpy as np
import torch
from typing import List, Tuple, Optional
from emotional_mi_pipeline import EmotionAnalysisPipeline


class AttentionWeights:
    """
    A class for calculating and preparing attention weight metrics from a transformer 
    model, specifically focusing on the attention paid to the token immediately 
    preceding the final prediction.
    """
    
    def __init__(self, mi_pipeline: EmotionAnalysisPipeline):
        
        self.mi_pipeline = mi_pipeline
        self.model = mi_pipeline.model
        self.tokenizer = mi_pipeline.tokenizer
        self.device = mi_pipeline.device
        
        

    def analyze_single_prompt_attn_wts(self, prompt: str, layer: int, head: int):
        """
        Generates and returns the raw attention weight matrix (query-key scores) 
        for a specific prompt, layer, and head using the TransformerLens framework.

        The returned matrix (T x T) shows the raw attention weights:
        - Rows: Query tokens (tokens receiving attention).
        - Columns: Key tokens (tokens being paid attention to).

        Args:
            prompt (str): The specific input text prompt to analyze.
            layer (int): The layer index (0 to N-1) to extract attention from.
            head (int): The head index (0 to N-1) within the specified layer.

        Returns:
            np.ndarray: The attention weight matrix (TokenLength x TokenLength) for 
                the specified layer and head, or None if extraction fails.
            """
        try:
            if self.model is None:
                print("Error: The model is not initialized (self.model is None)")
                return None
            
            # Run the model with cache using TransformerLens
            with torch.no_grad():
                
                _, cache = self.model.run_with_cache(prompt) 
            
            # Validation checks
            if layer >= self.model.cfg.n_layers:
                print(f"Error: Layer {layer} out of bounds for model with {self.model.cfg.n_layers} layers.")
                return None
            if head >= self.model.cfg.n_heads:
                print(f"Error: Head {head} out of bounds for model with {self.model.cfg.n_heads} heads.")
                return None
            
            layer_attn_tensor = cache["pattern", layer]
            
            head_attn_matrix = layer_attn_tensor[0, head, :, :]    #shape: (TokenLength, TokenLength)
            
            return head_attn_matrix.squeeze().detach().cpu().numpy()

        except Exception as e:
            print(f"Error Type: {type(e).__name__}, Error Message: {e}")
            return None
        
        
    def analyze_attention_contributions(self, prompt_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyzes a DataFrame of prompts by calculating the average attention weight 
        to the token immediately preceding the final prediction token, AVERAGED 
        ACROSS ALL PROMPTS IN THE INPUT DATAFRAME (SAMPLE).
        
        This metric measures how much a head focuses on the most recent contextual 
        information before the prediction.
        
        Args:
            prompt_df (pd.DataFrame): The DataFrame containing 'constrained prompt',
                                      'emotion', and 'predicted emotion' columns 
                                      (representing one sample).
                                      
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
            1. A DataFrame with the raw attention scores for every head/layer/prompt.
            2. A DataFrame with the average attention matrix statistics (Layer x Head) 
               for this sample.
        """
        all_attention_metrics = []
        
        for _, row in prompt_df.iterrows():
            
            prompt = row['constrained prompt']
            true_emotion = row['emotion']
            predicted_emotion = row['predicted emotion']
            
            try:
                
                # Defining the hook names for attention patterns
                hook_names = [f'blocks.{layer}.attn.hook_pattern' for layer in range(self.model.cfg.n_layers)]
                
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(
                        prompt, 
                        names_filter=hook_names, 
                        prepend_bos=True, 
                        device=self.model.cfg.device
                    )

                # final token index and previous token index
                pattern_layer_0 = cache[f'blocks.0.attn.hook_pattern'] # Using layer 0 to determine sequence length for all layers
                seq_len = pattern_layer_0.shape[-1]
                final_token_index = seq_len - 1
                
                if seq_len < 2:
                    print(f"Skipping attention analysis for short prompt: {prompt}")
                    continue
                    
                previous_token_index = final_token_index - 1


                for layer_idx in range(self.model.cfg.n_layers):
                    hook_name = f'blocks.{layer_idx}.attn.hook_pattern'
                    
                    # attention_patterns
                    attention_patterns = cache[hook_name].squeeze(0) 
                    
                    # attention_to_final_token_query has shape (num_heads, seq_len)
                    attention_to_final_token_query = attention_patterns[:, final_token_index, :] 

                    for head_idx in range(self.model.cfg.n_heads):
                        attn_to_previous_token = attention_to_final_token_query[head_idx, previous_token_index].item()
                        
                        all_attention_metrics.append({
                            'layer': layer_idx,
                            'head': head_idx,
                            'true_emotion': true_emotion,
                            'predicted_emotion': predicted_emotion,
                            'attention_to_previous_token': attn_to_previous_token
                        })
            except Exception as e:
                print(f"Skipping attention analysis for prompt: {row['constrained prompt']}\nError: {e}")
                continue

        if not all_attention_metrics:
            print("No attention metrics were generated. The prompt DataFrame might be empty or a model error occurred.")
            return pd.DataFrame(), pd.DataFrame()
            
        all_metrics_df = pd.DataFrame(all_attention_metrics)
        
        # Calculate the average attention metric for the entire input sample
        grouping_cols = ['layer', 'head'] 

        average_attention_df = all_metrics_df.groupby(grouping_cols).agg(
            avg_attention_mean=('attention_to_previous_token', 'mean'), 
            avg_attention_std=('attention_to_previous_token', 'std') 
        ).reset_index()

        return all_metrics_df, average_attention_df


    @staticmethod
    def average_attention_matrix_contributions(raw_metrics_df: pd.DataFrame) -> np.ndarray:
        """
        Converts the raw attention metrics DataFrame into the (Layer, Head) NumPy matrix 
        required for the differential heatmap.
        
        This method remains static as it only operates on the aggregated DataFrame 
        and does not require the model/tokenizer.

        Args:
            raw_metrics_df (pd.DataFrame): DataFrame containing 'layer', 'head', 
                                           and 'attention_to_previous_token' columns, 
                                           averaged across all desired prompts.

        Returns:
            np.ndarray: The averaged attention matrix of shape (num_layers, num_heads).
        """
        
        if raw_metrics_df.empty:
            
            return np.array([])
            
        
        # Grouping by layer and head to get the mean attention score
        avg_df = raw_metrics_df.groupby(['layer', 'head'])['attention_to_previous_token'].mean().reset_index()

        matrix_df = avg_df.pivot(index='layer', columns='head', values='attention_to_previous_token').fillna(0)
        
    
        return matrix_df.values
    
    
    def analyze_attn_baseline_contributions(self, baseline_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyzes a DataFrame of baseline (neutral) prompts by calculating the average 
        attention weight to the token immediately preceding the final prediction token, 
        AVERAGED ACROSS ALL BASELINE PROMPTS IN THE INPUT DATAFRAME (SAMPLE).
        
        This serves as a neutral baseline metric against which attention contributions 
        from emotion-constrained prompts can be compared.
        
        Args:
            baseline_df (pd.DataFrame): The DataFrame containing 'prompt' columns 
                                        for neutral sentences.
                                        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
            1. A DataFrame with the raw attention scores for every head/layer/prompt.
            2. A DataFrame with the average attention matrix statistics (Layer x Head) 
               for this baseline sample.
        """
        all_baseline_metrics = []
        
        for _, row in baseline_df.iterrows():
            
            prompt = row['constrained prompt']
            
            try:
            
                hook_names = [f'blocks.{layer}.attn.hook_pattern' for layer in range(self.model.cfg.n_layers)]
                
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(
                        prompt, 
                        names_filter=hook_names, 
                        prepend_bos=True, 
                        device=self.model.cfg.device
                    )

                pattern_layer_0 = cache[f'blocks.0.attn.hook_pattern']
                seq_len = pattern_layer_0.shape[-1]
                final_token_index = seq_len - 1
                
                if seq_len < 2:
                    print(f"Skipping baseline attention analysis for short prompt: {prompt}")
                    continue
                    
                previous_token_index = final_token_index - 1


                for layer_idx in range(self.model.cfg.n_layers):
                    
                    hook_name = f'blocks.{layer_idx}.attn.hook_pattern'

                    attention_patterns = cache[hook_name].squeeze(0) 
                
                    attention_to_final_token_query = attention_patterns[:, final_token_index, :] 

                    for head_idx in range(self.model.cfg.n_heads):

                        attn_to_previous_token = attention_to_final_token_query[head_idx, previous_token_index].item()
                        
                        all_baseline_metrics.append({
                            'layer': layer_idx,
                            'head': head_idx,
                            'prompt': prompt,
                            'attention_to_previous_token': attn_to_previous_token
                        })
            except Exception as e:
                print(f"Skipping baseline attention analysis for prompt: {prompt}\nError: {e}")
                continue

        if not all_baseline_metrics:
            
            print("No baseline attention metrics were generated")
            return pd.DataFrame(), pd.DataFrame()
            
        all_metrics_df = pd.DataFrame(all_baseline_metrics)
        
        grouping_cols = ['layer', 'head'] 

        average_baseline_attention_df = all_metrics_df.groupby(grouping_cols).agg(
            baseline_attention_mean=('attention_to_previous_token', 'mean'), 
            baseline_attention_std=('attention_to_previous_token', 'std') 
        ).reset_index()

        return all_metrics_df, average_baseline_attention_df    
    
    @staticmethod
    def top_attn_heads_contributions(differential_matrices: List[np.ndarray],k: int = 10) -> pd.DataFrame:
        """
        Identifies the top-k attention heads that show the largest average absolute 
        differential contribution across a list of differential matrices.

        This function quantifies which heads are most critical for the emotional steering 
        circuit by finding the heads with the highest average magnitude of change 
        (activation or suppression).

        Args:
            differential_matrices (List[np.ndarray]): A list of 32x32 NumPy arrays
                                                     where each matrix is 
                                                     (Constrained - Baseline).
            k (int): The number of top heads to return.

        Returns:
            pd.DataFrame: A DataFrame containing the top-k heads, ranked by their
            average absolute differential score.
        """
        if not differential_matrices:
            return pd.DataFrame()

        # Flattening all matrices and convert to absolute scores
        data = []
        for matrix in differential_matrices:
            for layer in range(matrix.shape[0]):
                for head in range(matrix.shape[1]):
                    data.append({
                        'layer': layer,
                        'head': head,
                        'abs_diff_score': np.abs(matrix[layer, head])
                    })

        df = pd.DataFrame(data)

        # Average the absolute scores across all differential conditions
        avg_df = df.groupby(['layer', 'head'])['abs_diff_score'].mean().reset_index()
        avg_df.rename(columns={'abs_diff_score': 'average_abs_contribution'}, inplace=True)

        #Rank and return the top-k heads
        top_heads_df = avg_df.sort_values(
            by='average_abs_contribution',
            ascending=False
        ).head(k).reset_index(drop=True)
        
        return top_heads_df

    @staticmethod
    def top_activating_suppressing_heads(differential_matrices: List[np.ndarray], k: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identifies the top-k attention heads that show the largest average POSITIVE (activating)
        and NEGATIVE (suppressing) differential contributions across a list of differential
        matrices.

        This quantifies which heads are most consistently activated (positive change)
        or suppressed (negative change) when moving from a baseline to a constrained
        emotional state.

        Args:
            differential_matrices (List[np.ndarray]): A list of NumPy arrays
                                                     where each matrix is (Constrained - Baseline).
            k (int): The number of top heads to return for both activation and suppression.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
            1. DataFrame: Top-k activating heads (highest mean positive score).
            2. DataFrame: Top-k suppressing heads (lowest mean negative score).
        """
        if not differential_matrices:
            return pd.DataFrame(), pd.DataFrame()

        data = []
        for matrix in differential_matrices:
            for layer in range(matrix.shape[0]):
                for head in range(matrix.shape[1]):
                    data.append({
                        'layer': layer,
                        'head': head,
                        'raw_diff_score': matrix[layer, head]
                    })

        df = pd.DataFrame(data)

        avg_df = df.groupby(['layer', 'head'])['raw_diff_score'].mean().reset_index()
        avg_df.rename(columns={'raw_diff_score': 'average_differential_score'}, inplace=True)

        top_activating_df = avg_df.sort_values(
            by='average_differential_score',
            ascending=False
        ).head(k).reset_index(drop=True)
        top_activating_df['type'] = 'Activating'

        top_suppressing_df = avg_df.sort_values(
            by='average_differential_score',
            ascending=True
        ).head(k).reset_index(drop=True)
        top_suppressing_df['type'] = 'Suppressing'

        return top_activating_df, top_suppressing_df
