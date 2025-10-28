import pandas as pd
import numpy as np
import torch
from typing import Tuple
from emotional_mi_pipeline import EmotionAnalysisPipeline


class AttentionWeights:
    """
    A class for calculating and preparing attention weight metrics from a transformer 
    model, specifically focusing on the attention paid to the token immediately 
    preceding the final prediction.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initializes the AttentionWeights analyzer with the required model and tokenizer.
        
        Args:
            model: The transformer model instance (e.g., HookedTransformer).
            tokenizer: The tokenizer object.
        """
        self.model = model
        self.tokenizer = tokenizer

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
        
        # Iterate over each prompt in the sample DataFrame
        for _, row in prompt_df.iterrows():
            
            prompt = row['constrained prompt']
            true_emotion = row['emotion']
            predicted_emotion = row['predicted emotion']
            
            try:
                # 1. Run model and cache attention patterns
                # Define the hook names for attention patterns
                hook_names = [f'blocks.{layer}.attn.hook_pattern' for layer in range(self.model.cfg.n_layers)]
                
                with torch.no_grad():
                    # We only need the cache for attention analysis
                    # Running the model with cache to get attention patterns
                    _, cache = self.model.run_with_cache(
                        prompt, 
                        names_filter=hook_names, 
                        prepend_bos=True, 
                        device=self.model.cfg.device
                    )

                # 2. Determine final token index and previous token index
                # Use layer 0 to determine sequence length for all layers
                pattern_layer_0 = cache[f'blocks.0.attn.hook_pattern']
                seq_len = pattern_layer_0.shape[-1]
                final_token_index = seq_len - 1
                
                if seq_len < 2:
                     print(f"Skipping attention analysis for short prompt: {prompt}")
                     continue
                     
                previous_token_index = final_token_index - 1


                # 3. Iterate over layers and heads to extract the metric
                for layer_idx in range(self.model.cfg.n_layers):
                    hook_name = f'blocks.{layer_idx}.attn.hook_pattern'
                    
                    # attention_patterns shape is (1, num_heads, seq_len, seq_len). Squeeze batch dim.
                    attention_patterns = cache[hook_name].squeeze(0) 
                    
                    # attention_to_final_token_query has shape (num_heads, seq_len)
                    # Row is the attention weight placed by the final query token on all source tokens.
                    attention_to_final_token_query = attention_patterns[:, final_token_index, :] 

                    for head_idx in range(self.model.cfg.n_heads):
                        # Metric: Attention paid by the final query token 
                        # to the previous token (index = previous_token_index)
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
        
        # Calculate the average attention metric for the ENTIRE input sample
        grouping_cols = ['layer', 'head'] 

        average_attention_df = all_metrics_df.groupby(grouping_cols).agg(
            avg_attention_mean=('attention_to_previous_token', 'mean'), 
            avg_attention_std=('attention_to_previous_token', 'std') 
        ).reset_index()

        # The returned average_attention_df is the average L x H data for this specific sample.
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
            # Return an empty array if no metrics are present
            return np.array([])
            
        # Calculate mean if the input is the raw, unaggregated metrics (for safety)
        # Grouping by layer and head to get the mean attention score
        avg_df = raw_metrics_df.groupby(['layer', 'head'])['attention_to_previous_token'].mean().reset_index()

        # Pivot the data into the desired (Layer, Head) matrix format.
        # The 'attention_to_previous_token' value becomes the cell data.
        matrix_df = avg_df.pivot(index='layer', columns='head', values='attention_to_previous_token').fillna(0)
        
        # Convert to NumPy array
        return matrix_df.values