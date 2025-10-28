import torch
import pandas as pd
from transformer_lens import HookedTransformer
from huggingface_hub import login
from typing import List, Tuple, Dict, Any
import torch.nn.functional as F
import numpy as np
import sys
import re
import random


class EmotionAnalysisPipeline:
    """
    A pipeline for zero-shot emotion classification using a Llama2 model,
    with built-in utilities for Logit Lens and Causal Tracing preparation.
    
    The classification is performed using deterministic logit-filtering 
    on a constrained set of emotion tokens.
    """
    def __init__(self, model_name: str, hf_token: str, device: str = "cuda"):
        """
        Initializes the model, tokenizer, and device, and sets up the target emotion tokens.
        
        Args:
            model_name (str): The name of the Hugging Face model (e.g., 'meta-llama/Llama-2-7b-hf').
            hf_token (str): The Hugging Face authentication token.
            device (str): The device to run the model on ('cuda' or 'cpu').
        
        Raises:
            ValueError: If the emotion token setup fails to find unique token IDs.
        """
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Standard emotion labels mapping
        self.emotion_labels = {
            'sad': 0, 'joy': 1, 'love': 2, 'anger': 3,
            'fear': 4, 'surprise': 5
        }
        
        self._load_model(model_name, hf_token)
        
        self.emotion_ids: List[int]
        self.id_to_emotion_map: Dict[int, str]
        self.emotion_token_tensor: torch.Tensor
        
        self.emotion_ids, self.id_to_emotion_map, self.emotion_token_tensor = self._setup_emotion_tokens()
        
        print(f"Target Emotion Token IDs: {self.emotion_ids}")
        print(f"Target Token Mappings (ID: Emotion): {self.id_to_emotion_map}")


    def _load_model(self, model_name: str, hf_token: str):
        """Logs in to Hugging Face and loads the pre-trained model using HookedTransformer."""
        try:
            print("Logging into Hugging Face...")
            login(token=hf_token)
            print(f"Loading model: {model_name}...")
           
            self.model = HookedTransformer.from_pretrained(
                model_name,
                fold_ln=False,
                center_unembed=False,
                center_writing_weights=False,
                device=self.device
            )
            self.tokenizer = self.model.tokenizer
            print(f"Model and tokenizer loaded successfully on device: {self.device}")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}", file=sys.stderr)
            raise
    
    def _setup_emotion_tokens(self) -> Tuple[List[int], Dict[int, str], torch.Tensor]:
        """
        Creates a mapping by extracting the most contextually relevant emotion token ID 
        for each label (usually the second token after a space prefix for Llama). 
        Prepares the token tensor for logit filtering.
        
        Returns:
            Tuple[List[int], dict, torch.Tensor]: List of IDs, ID-to-emotion map, and ID tensor.
        """
        emotion_label_list = list(self.emotion_labels.keys())
        emotion_token_map: Dict[int, str] = {}
        target_token_ids: List[int] = [] 
        
        for label in emotion_label_list:
            try:
                # Get the full list of tokens for the spaced emotion word (e.g., ' joy' -> [ID_SPACE, ID_JOY])
                token_ids = self.get_token_ids(label) 

                # For Llama, the second token (index 1) usually represents the core word.
                if len(token_ids) >= 2:
                    token_id = token_ids[1] 
                elif len(token_ids) == 1:
                    token_id = token_ids[0]
                    print(f"Warning: '{label}' tokenized into one token: {token_id}. Using it.", file=sys.stderr)
                else:
                     raise ValueError("No tokens found.")
                     
                if token_id in target_token_ids:
                    raise ValueError(f"FATAL: Token ID {token_id} for '{label}' is a duplicate.")

                target_token_ids.append(token_id)
                emotion_token_map[token_id] = label
                
            except ValueError as e:
                print(f"Skipping emotion '{label}' due to tokenization issue: {e}", file=sys.stderr)

        if not target_token_ids:
            raise ValueError("Failed to setup any unique emotion token IDs")

        emotion_token_tensor = torch.tensor(target_token_ids, device=self.device)
        return target_token_ids, emotion_token_map, emotion_token_tensor

    def _generate_prompt(self, text: str) -> str:
        """
        Generates a constraint-based prompt for emotion classification using short labels.
        
        Args:
            text (str): The input text to be classified.
            
        Returns:
            str: The fully formatted, constrained prompt string.
        """
        emotions_str = ', '.join(self.emotion_labels.keys())
        return (f"{text}. What is the single emotion of this text? "
                f"You must choose one and only one from the following list: {emotions_str}. "
                "The emotion is:")

    def classify_emotion_logit_based(self, text: str) -> str:
        """
        Classifies the emotion of a given text using logit-based constrained prompting.
        It restricts the output generation to only the pre-defined emotion tokens.
        
        Args:
            text (str): The input text to classify.
            
        Returns:
            str: The predicted emotion label.
        """
        constraint_prompt = self._generate_prompt(text)
        
        inputs_constraint = self.tokenizer(constraint_prompt, return_tensors="pt")
        input_ids_constraint = inputs_constraint['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs_constraint = self.model(input_ids_constraint)
            
            # Access raw logits tensor
            raw_logits = outputs_constraint[0] if isinstance(outputs_constraint, tuple) else outputs_constraint
            
            # Access the logits for the last token position
            next_token_logits_constraint = raw_logits[:, -1, :] 
            
            # Filter the logits to include only the designated emotion tokens
            filtered_logits_constraint = next_token_logits_constraint.index_select(
                1, self.emotion_token_tensor
            ).squeeze(0)

            # Find the index of the highest logit *within the filtered set*
            predicted_index_constraint = torch.argmax(filtered_logits_constraint).item() 
            predicted_token_id_constraint = self.emotion_ids[predicted_index_constraint]
            
            predicted_emotion = self.id_to_emotion_map.get(predicted_token_id_constraint, "unknown_error")
            return predicted_emotion


    def get_token_ids(self, text: str) -> List[int]:
        """
        Gets the token IDs for a text string, prioritizing tokenization with a leading space.
        
        Args:
            text (str): The string (e.g., 'joy', 'anger') to tokenize.
            
        Returns:
            List[int]: A list of token IDs.
            
        Raises:
            ValueError: If no tokens can be found for the text.
        """
        try:
            # Try tokenizing with a leading space (standard for subsequent words)
            encoded_with_space = self.tokenizer.encode(f" {text}", add_special_tokens=False)
            if encoded_with_space:
                return encoded_with_space
            # Fallback to tokenizing without a space
            encoded_without_space = self.tokenizer.encode(text, add_special_tokens=False)
            if encoded_without_space:
                return encoded_without_space
        except Exception as e:
            print(f"Tokenization error for '{text}': {e}", file=sys.stderr)
            
        raise ValueError(f"Could not find any tokens for '{text}'.")
    

    def get_unique_token_id(self, text: str) -> int:
        """
        Gets the *final* token ID for a text string intended for logit analysis.
        
        This is crucial for BPE models (like Llama) where a word often tokenizes into 
        [SPACE_TOKEN, WORD_ROOT]. This function returns only the final token ID (WORD_ROOT)
        to isolate the word's primary signal for tracing purposes.
        
        Args:
            text (str): The emotion word (e.g., 'love', 'joy').
            
        Returns:
            int: The single, unique token ID.
        
        Raises:
            ValueError: If no tokens can be found for the text.
        """
        text_to_tokenize = text.strip()
        
        try:
            # Try with leading space
            encoded_with_space = self.tokenizer.encode(f" {text_to_tokenize}", add_special_tokens=False)
            
            if encoded_with_space:
                return encoded_with_space[-1]
            
            # Fallback without leading space
            encoded_without_space = self.tokenizer.encode(text_to_tokenize, add_special_tokens=False)
            if encoded_without_space:
                return encoded_without_space[-1]

        except Exception as e:
            print(f"Tokenization error for '{text}': {e}", file=sys.stderr)
            
        raise ValueError(f"Could not find any tokens for '{text}'.")
    

    def get_rank(self, logits: torch.Tensor, token_ids: List[int]) -> int:
        """
        Calculates the minimum 1-based rank of the given token_ids 
        in the full logits tensor using vectorized operations.

        Args:
            logits (torch.Tensor): The output logits tensor (size: [Vocabulary Size]).
            token_ids (List[int]): A list of target token IDs whose ranks should be found.

        Returns:
            int: The minimum 1-based rank (1 is the highest rank) among the target tokens.
        """
        if not token_ids:
            return len(logits) + 1

        # Sorts the logits descendingly to get the token IDs in rank order (best to worst).
        sorted_token_ids = torch.argsort(logits, descending=True)

        # Creates a map where index is the token ID and value is the 0-based rank
        rank_map = torch.empty_like(sorted_token_ids)
        rank_map[sorted_token_ids] = torch.arange(len(logits), device=logits.device)

        # Get the 0-based ranks for the target tokens
        target_token_tensor = torch.tensor(token_ids, device=logits.device)
        zero_based_ranks = rank_map[target_token_tensor]

        # Find the best (minimum) rank
        min_zero_based_rank = torch.min(zero_based_ranks).item()
        
        # Convert to 1-based rank (Rank 0 -> Rank 1)
        return min_zero_based_rank + 1

    def calculate_prompt_ranks(self, prompt_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes a DataFrame of prompts by calculating the minimum rank for the true emotion
        across all layers, based on MLP residual stream contributions (a form of Logit Lens).
        
        This method is used to identify the layer where the correct emotion signal first appears.

        Args:
            prompt_df (pd.DataFrame): DataFrame containing 'emotion' and 'constrained prompt' columns.

        Returns:
            pd.DataFrame: The original DataFrame augmented with columns for the 
                          minimum true rank across all layers and the layer where it occurred.
        """
        categorized_prompts = []
        for index, row in prompt_df.iterrows():
            try:
                true_emotion_text = row['emotion']
                prompt_text = row['constrained prompt']
                true_ids = self.get_token_ids(true_emotion_text)
                input_ids = self.tokenizer.encode(prompt_text, return_tensors='pt')
                final_token_idx = input_ids.shape[-1] - 1
                
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(prompt_text)

                min_true_rank_across_layers = float('inf')
                min_rank_layer_idx = -1
                
                # Iterate through all layers to find the best MLP-only rank
                for layer_idx in range(self.model.cfg.n_layers):
                    # Residual stream contribution from the MLP 
                    mlp_out_contribution = cache[("mlp_out", layer_idx)][0, final_token_idx, :]
                    mlp_logits = self.model.unembed(mlp_out_contribution)
                    true_rank = self.get_rank(mlp_logits, true_ids)
                    
                    if true_rank < min_true_rank_across_layers:
                        min_true_rank_across_layers = true_rank
                        min_rank_layer_idx = layer_idx
                        
                # Rank at the final residual stream (full model output)
                final_residual_rank = self.get_rank(
                    self.model.unembed(cache['resid_post', self.model.cfg.n_layers - 1][0, final_token_idx, :]), 
                    true_ids
                )
                        
                categorized_prompts.append({
                    'prompt': prompt_text,
                    'rank last layer': final_residual_rank,
                    'min_true_rank': min_true_rank_across_layers,
                    'min_rank_layer': min_rank_layer_idx
                })
            except Exception as e:
                print(f"Skipping analysis for row {index}. Error: {e}", file=sys.stderr)
                continue

        if not categorized_prompts:
            return pd.DataFrame()
            
        return pd.DataFrame(categorized_prompts)

    def categorize_prompts(self, ranked_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the median of 'min_true_rank' as a knowledge threshold and
        categorizes prompts into 'Extraction' or 'Enrichment' hallucination types.
        
        - 'Extraction': Correct emotion signal appears early (rank <= median).
        - 'Enrichment': Correct emotion signal appears late (rank > median), suggesting 
                        late-stage knowledge injection or error accumulation.
                        
        Args:
            ranked_df (pd.DataFrame): DataFrame output from calculate_prompt_ranks.
            
        Returns:
            pd.DataFrame: DataFrame augmented with 'threshold' and 'hallucination type'.
        """
        if ranked_df.empty:
            print("Input DataFrame is empty. Cannot categorize prompts.", file=sys.stderr)
            return ranked_df

        try:
            knowledge_threshold = np.median(ranked_df['min_true_rank'])
        except KeyError as e:
            print(f"DataFrame is missing the required column: {e}", file=sys.stderr)
            return ranked_df

        ranked_df['threshold'] = knowledge_threshold
        ranked_df['hallucination type'] = ranked_df['min_true_rank'].apply(
            lambda min_rank: "Extraction" if min_rank <= knowledge_threshold else "Enrichment"
        )
        return ranked_df
