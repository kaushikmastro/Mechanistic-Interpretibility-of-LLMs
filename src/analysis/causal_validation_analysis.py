import pandas as pd
import torch
import torch.nn.functional as F 
import numpy as np 
import sys 
import re
import random
from typing import List, Tuple, Dict, Any



class CausalValidationAnalysis:
    
    def __init__(self, mi_pipeline):
        
        self.mi_pipeline = mi_pipeline
        self.model = mi_pipeline.model
        self.tokenizer = mi_pipeline.tokenizer
        self.device = mi_pipeline.device


    def get_model_embedding_layer(self):
        """
        Finds and returns the model's token embedding layer, handling various attribute names.
        """
        if hasattr(self.model, 'get_input_embeddings'):
            return self.model.get_input_embeddings()
        elif hasattr(self.model, 'embed'):
            return self.model.embed
        elif hasattr(self.model, 'w_e'):
            return self.model.w_e
        
       
        if hasattr(self.model.config, 'vocab_size') and hasattr(self.model.config, 'hidden_size'):
            print("Warning: Standard embedding layer access failed. Assuming model has a standard 'wte' for token embeddings.", file=sys.stderr)
            return self.model.wte if hasattr(self.model, 'wte') else None
        
        raise AttributeError("Could not find the model's input embedding layer.")


    def calculate_calibration_stats(self, prompts: List[str]) -> Tuple[float, torch.Tensor]:
        """
        Calculates sigma (3 * empirical std dev) and the mean embedding of a set of prompts.
        
        Args:
            prompts (List[str]): A list of prompts to calibrate on.

        Returns:
            Tuple[float, torch.Tensor]: A tuple containing the sigma value and the mean embedding tensor.
        """
        
        all_embeddings = []
        try:
            embedding_module = self.get_model_embedding_layer()
        except AttributeError as e:
            print(f"Error getting embedding layer: {e}", file=sys.stderr)
            return 0.0, torch.zeros(self.model.config.hidden_size, device=self.device) 
            
        device = self.device

        
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            with torch.no_grad():
                # using the returned embedding module for the forward pass
                token_embeddings = embedding_module(input_ids)
            
                # mean embedding across all tokens in the sequence
                mean_embedding = torch.mean(token_embeddings, dim=1)
                all_embeddings.append(mean_embedding.squeeze())
            
        if not all_embeddings:
            embedding_dim = self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else 768 
            return 0.0, torch.zeros(embedding_dim, device=device) # Safely return zero tensor based on the model's hidden size

        stacked_embeddings = torch.stack(all_embeddings)

        # standard deviation and mean of all embeddings
        empirical_std_dev = torch.std(stacked_embeddings, dim=0, unbiased=False)
        mean_embedding = torch.mean(stacked_embeddings, dim=0)

        # sigma as 3 times the mean of the std dev vector
        sigma = 3 * torch.mean(empirical_std_dev).item()
        
        return sigma, mean_embedding

    @staticmethod
    def get_positive_LLR_prompts(df: pd.DataFrame, target_layer) -> pd.DataFrame:
        """
        Filters the causal tracing DataFrame to return only rows that meet two criteria:
        1. The 'layer' is equal to the target layer (default: 31).
        2. The 'log_likelihood_ratio' (or 'logit_difference') is positive (> 0).

        Args:
            df: The input DataFrame containing all causal tracing results across layers.

        Returns:
            A new DataFrame containing only the filtered rows and the required columns.
        """
        
        df_temp = df.copy()
        if 'log_likelihood_ratio' not in df_temp.columns:
            if 'logit_difference' in df_temp.columns:
                df_temp = df_temp.rename(columns={'logit_difference': 'log_likelihood_ratio'})
            else:
                print("Error: DataFrame must contain 'log_likelihood_ratio' or 'logit_difference' column.", file=sys.stderr)
                return pd.DataFrame()

        df_filtered_layer = df_temp[df_temp['layer'] == target_layer].copy()

        #positive log likelihood ratios (y > 0)
        df_positive_ratio = df_filtered_layer[df_filtered_layer['log_likelihood_ratio'] > 0].copy()

        required_cols = [
            'prompt_text',
            'true_emotion',
            'predicted_emotion',
            'layer',
            'log_likelihood_ratio'
            ]

        final_df = df_positive_ratio.filter(items=required_cols)

        return final_df

    def get_subject_token_and_index(self, prompt: str):
        """
        Identifies the subject token (assumed to be the first token after cleaning)
        and finds its corresponding index in the tokenized sequence.
        
        Args:
            prompt (str): The original prompt text.
        
        Returns:
            Optional[Tuple[str, int]]: A tuple containing the subject token text and 
            its index, or None if the prompt is invalid.
        """
        # Cleaning the prompt by stripping leading/trailing quotes and spaces
        clean_prompt = prompt.strip(' \'"')
        
        # Tokenize the cleaned prompt
        tokenized_output = self.tokenizer(clean_prompt, return_tensors='pt', add_special_tokens=True)
        input_ids = tokenized_output.input_ids[0].tolist()
        
        if len(input_ids) < 1: 
            return None
        
        # Get the set of all special token IDs for a robust check
        special_ids = set(self.tokenizer.all_special_ids)
        
        subject_index = 0
        subject_text = ""
        
        # 3. Iterate through token IDs to find the first non-special, non-empty token
        for i, token_id in enumerate(input_ids):
            if token_id not in special_ids:
            # Decode to check if it's a non-empty content token
                token_text = self.tokenizer.decode(token_id, skip_special_tokens=False)
            
            # We check that the token text isn't just an empty string or a space-only token
                if token_text.strip() != "":
                    subject_index = i
                    subject_text = token_text.strip()
                    break
        
        if not subject_text:
            return None
        
        # 4. Return the first subject token and its index
        return subject_text, subject_index

    def calculate_ll_from_embeddings(self, u_star_embeddings: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-likelihood (LL) of the *first* target answer token 
        given a starting embedding tensor (u*). This is used as the proxy metric 
        (y') for the Noise Injection stage.

        This function uses the highly robust 'run_with_hooks' method to inject 
        the custom u* embeddings.

        Args:
            u_star_embeddings (torch.Tensor): The perturbed embedding tensor (u*). 
                                             Shape: [1, seq_len, d_model].
            target_ids (torch.Tensor): The token IDs of the target answer sequence (o or o'). 
                                       Shape: [1, target_seq_len].

        Returns:
            torch.Tensor: The log-likelihood of the first target token (single-element tensor).
        """
        
        prompt_seq_len = u_star_embeddings.shape[1]
        
        # Create dummy input tokens. The values don't matter because the hook will 
        # replace the embeddings before any computation occurs.
        dummy_tokens = torch.zeros((1, prompt_seq_len), 
                                   dtype=torch.long, 
                                   device=u_star_embeddings.device)

        # 1. Define the hook function
        def embed_replacer(resid_pre: torch.Tensor, hook) -> torch.Tensor:
            """Hook to replace the initial residual stream input with u* embeddings."""
            return u_star_embeddings
        
        # 2. Define the hook point (the input to the residual stream before block 0)
        # NOTE: This assumes TransformerLens compatibility or a custom method in self.model
        hook_name = 'blocks.0.hook_resid_pre' 
        
        # 3. Run the model with the hook
        # Logits shape: [1, seq_len, vocab_size]
        with torch.no_grad():
            if not hasattr(self.model, 'run_with_hooks'):
                 # Fallback/Error if not HookedTransformer - adjust hook name or architecture access here
                print("Error: Model does not have 'run_with_hooks'. This feature relies on TransformerLens or equivalent.", file=sys.stderr)
                return torch.tensor([-1000.0], device=u_star_embeddings.device) 
                
            logits = self.model.run_with_hooks(
                dummy_tokens,
                fwd_hooks=[(hook_name, embed_replacer)]
            )
        
        # 4. Calculate the log probability of the first target token
        last_prompt_logits = logits[0, prompt_seq_len - 1, :]
        
        # Apply log-softmax for stable log probabilities
        last_prompt_log_probs = F.log_softmax(last_prompt_logits, dim=-1)
        
        # Get the ID of the first true answer token
        if target_ids.numel() == 0:
        # Return a tensor with a very low LL if the target is empty.
            return torch.tensor([-1000.0], device=u_star_embeddings.device)
            
        first_target_id = target_ids[0, 0].item()
        
        # Extract the log probability
        ll_first_token_tensor = last_prompt_log_probs[first_target_id].unsqueeze(0)

        return ll_first_token_tensor

    def perform_causal_analysis_sub_token(self, sample_df: pd.DataFrame, sigma_value: float, num_noise_samples: int) -> pd.DataFrame:
        """
        Performs the first stage of Causal Tracing (Noise Injection).
        
        The filtered truthful hidden states (u*), their corresponding y' scores,
        and the intervention site details are collected. The resulting DataFrame
        RETAINS all original columns.

        Args:
            sample_df (pd.DataFrame): DataFrame containing prompts, true answers (o), 
                                     and hallucinated answers (o').
            sigma_value (float): The calculated standard deviation for the noise.
            num_noise_samples (int): The number of noise samples to generate per prompt.

        Returns:
            pd.DataFrame: A new DataFrame containing ALL original columns, plus 
                          the intervention details, sample count, hidden states, 
                          and y' scores.
        """
        
        device = self.device
        
        print(f"Performing Causal Tracing (Noise Injection) on device: {device}", file=sys.stderr)
        self.model.to(device)

        results_data = []
        # Intervention layer is 0, since we are injecting noise into the embedding (input) layer
        intervention_layer = 0 

        self.model.eval()
        
        with torch.no_grad():
            for index, row in sample_df.iterrows():
                prompt = row['prompt_text']
                
                # FIX: Corrected method call
                subject_info = self.get_subject_token_and_index(prompt)
                
                if subject_info is None:
                    print(f"Skipping row {index}: Could not identify subject token for prompt: '{prompt}'", file=sys.stderr)
                    continue
                
                subject_token, subject_token_index = subject_info
                
                # Tokenization and ID setup...
                prompt_stripped = prompt.strip(' \'"')
                tokenized_input = self.tokenizer(prompt_stripped, return_tensors='pt', add_special_tokens=True)
                input_ids = tokenized_input.input_ids.to(device)
                if input_ids.numel() == 0: continue

                # FIX: Corrected method call
                original_embeddings = self.get_model_embedding_layer()(input_ids)
                
                # Get target IDs (o' is the TRUE emotion, o is the PREDICTED/HALLUCINATED emotion)
                o_prime_id = self.mi_pipeline.get_unique_token_id(row['true_emotion'])
                o_id = self.mi_pipeline.get_unique_token_id(row['predicted_emotion'])

                o_ids = torch.tensor([[o_id]], dtype=torch.long).to(device)
                o_prime_ids = torch.tensor([[o_prime_id]], dtype=torch.long).to(device)

                if o_ids.numel() == 0 or o_prime_ids.numel() == 0 or o_id == -1 or o_prime_id == -1: 
                    print(f"Skipping row {index}: Could not tokenize target emotions.", file=sys.stderr)
                    continue
                
                truthful_hidden_states = []
                truthful_y_primes = [] 
                
                for _ in range(num_noise_samples):
                
                # 1. Generate noise
                    noise = torch.randn_like(original_embeddings).to(device) * sigma_value
                    u_star_embeddings = original_embeddings.clone()
                
                # 2. Perturb only at the subject token index (i.e., [0, subject_token_index, :])
                    u_star_embeddings[0, subject_token_index, :] += noise[0, subject_token_index, :]
                
                    try:
                    # 3. Calculate LLs
                        # FIX: Corrected method call
                        ll_o_prime_u_star = self.calculate_ll_from_embeddings(u_star_embeddings, o_prime_ids)
                        # FIX: Corrected method call
                        ll_o_u_star = self.calculate_ll_from_embeddings(u_star_embeddings, o_ids)

                    # 4. Calculate y' (LLR: Hallucinated LL - True LL)
                    # A negative y' means the noise successfully switched the model to predict the *true* emotion (o').
                        y_prime = (ll_o_u_star - ll_o_prime_u_star).item()
                    
                        if y_prime < 0:
                        # 5. Store the successful (truth-inducing) hidden state (u*) and its score (y')
                        # We store the *perturbed* embedding u*
                            truthful_hidden_states.append(u_star_embeddings.cpu().squeeze(0).clone())
                            truthful_y_primes.append(y_prime)
                        
                    except Exception as e:
                        print(f"Error during LL calculation for row {index}: {e}", file=sys.stderr)

            # Store results for the current prompt
                current_row = row.to_dict() 
                
                current_row['intervention_layer'] = intervention_layer
                current_row['subject_token'] = subject_token
                current_row['token_index'] = subject_token_index
                current_row['num_truth_inducing_samples'] = len(truthful_hidden_states)
                current_row['truthful_hidden_states'] = truthful_hidden_states
                current_row['truthful_y_primes'] = truthful_y_primes
                
                results_data.append(current_row)

        # Create and return the final results DataFrame
        final_df = pd.DataFrame(results_data)
        
        return final_df

    @staticmethod
    def generalize_truth_results_sub_token(truth_results_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregates and filters multiple causal tracing DataFrames, categorizes prompts,
        and returns a clean, concise result set, excluding detailed intervention columns.

        Args:
            truth_results_dict: A dictionary where keys are hallucination types (e.g., 'anger-joy')
                                and values are the pandas DataFrames from the causal tracing stage.

        Returns:
            A single pandas DataFrame containing only the successful interventions,
            with prompts categorized and detailed intervention columns removed.
        """
        all_successful_results = []

        # 1. Calculate the global median rank (from all prompts combined)
        try:
            all_dfs = pd.concat([df for df in truth_results_dict.values() if not df.empty])
            
            # Check if the concatenated DataFrame is empty
            if all_dfs.empty:
                print("Error: Input dictionary contains no valid dataframes for aggregation.")
                return pd.DataFrame()
            
            # The log_likelihood_ratio (y_corrupt) is used as the proxy for difficulty/type
            median_rank = all_dfs['log_likelihood_ratio'].median()
        except ValueError as e:
            print(f"Error during median calculation: {e}. Returning empty DataFrame.")
            return pd.DataFrame()

        # 2. Define the columns to keep in the final output
        keep_cols = [
            'prompt_text',
            'log_likelihood_ratio',
            'true_emotion',
            'predicted_emotion',
            'num_truth_inducing_samples',
            'truthful_hidden_states',
            'truthful_y_primes'
        ]

        for key, df in truth_results_dict.items():
            if df.empty:
                continue
            
            # Filter for successful interventions (only rows where truth was successfully restored)
            successful_df = df[df['num_truth_inducing_samples'] > 0].copy()

            if successful_df.empty:
                continue

            # 3. Categorize prompts based on the median rank of the corrupted prediction
            def categorize_prompt(y_corrupt):
                """
                Categorizes the prompt based on how confidently the model made the incorrect prediction.
                Lower y_corrupt (higher confidence in the *corrupt* prediction) suggests a strong,
                retrieved 'fact' (Extraction).
                """
                if y_corrupt <= median_rank:
                    return 'Extraction' # Strong, fact-like misbehavior
                else:
                    return 'Enrichment' # Weaker, more reasoning-like misbehavior

            successful_df['category'] = successful_df['log_likelihood_ratio'].apply(categorize_prompt)
            successful_df['hallucination_type'] = key # Add the type of hallucination for analysis

            # 4. Filter the dataframe to only include the required columns
            cols_to_select = [col for col in keep_cols if col in successful_df.columns]
            cols_to_select.extend(['category', 'hallucination_type'])
            
            all_successful_results.append(successful_df[cols_to_select])

        # 5. Concatenate all results into a single, clean DataFrame
        if not all_successful_results:
            return pd.DataFrame()
            
        final_df = pd.concat(all_successful_results, ignore_index=True)
        return final_df


    def get_random_token_index(self, prompt: str):
        
        """Finds a random token index from the prompt to inject noise (0 to seq_len-2)."""
        
        # FIX: Corrected typo (selftokenizer -> self.tokenizer)
        # Tokenize without special tokens to get indices corresponding to the prompt content
        input_ids = self.tokenizer.encode(prompt.strip(' \'"'), add_special_tokens=False)
        if not input_ids or len(input_ids) < 2:
            return None
        # Exclude the last token (completion token)
        return random.randint(0, len(input_ids) - 2) 


    def perform_causal_analysis_baseline(self, prompt_df, sigma_value, num_noise_samples):
        """
        Performs causal analysis by adding scaled Gaussian noise to the embeddings of
        randomly chosen tokens (including the potential causal token) as a baseline check.
        
        Args:
            prompt_df (pd.DataFrame): DataFrame with prompts and emotion targets.
            sigma_value (float): The calibration factor for noise magnitude.
            num_noise_samples (int): The number of noise samples per prompt.

        Returns:
            pd.DataFrame: DataFrame summarizing the baseline noise results.
        """
        results_list: List[Dict[str, Any]] = []
        
        self.model.eval()
        device = self.device
        
        with torch.no_grad():
            
            for index, row in prompt_df.iterrows():
                
                prompt_to_analyze = row['prompt_text'] 
                true_emotion = row['true_emotion']
                predicted_emotion = row['predicted_emotion']
                
                # FIX: Added np check
                llr = row.get('log_likelihood_ratio', row.get('logit_difference', np.nan))
                
                # 1. Setup: Get a random index from the available tokens
                # FIX: Corrected method call (removed tokenizer arg, added self.)
                random_token_idx = self.get_random_token_index(prompt_to_analyze)
                
                prompt_stripped = prompt_to_analyze.strip(' \'"')
                # FIX: Corrected global function call
                input_ids_content = self.tokenizer.encode(prompt_stripped, add_special_tokens=False) 
                
                if random_token_idx is None:
                    continue
                
                # Decode the token name for the output report
                # FIX: Corrected global function call
                noise_token = self.tokenizer.decode([input_ids_content[random_token_idx]]) 
                
                truthful_y_primes = []
                
                # Get the unique ID for the Predicted Emotion (o)
                # FIX: Corrected global function call (should use mi_pipeline method)
                o_id = self.mi_pipeline.get_unique_token_id(predicted_emotion)
                o_target_ids = torch.tensor([[o_id]], dtype=torch.long).to(device)

                # Get the unique ID for the True Emotion (o')
                # FIX: Corrected global function call (should use mi_pipeline method)
                o_prime_id = self.mi_pipeline.get_unique_token_id(true_emotion)
                o_prime_target_ids = torch.tensor([[o_prime_id]], dtype=torch.long).to(device)

                if o_id == -1 or o_prime_id == -1: continue
                
                # 2. Get the original embeddings (u)
                tokenized_input = self.tokenizer(prompt_stripped, return_tensors='pt', add_special_tokens=True).input_ids.to(device)
                # FIX: Corrected method call
                original_embeddings = self.get_model_embedding_layer()(tokenized_input)
                
                for _ in range(num_noise_samples):
                # 3. Generate noise (u*) and perturb the embeddings
                    noise = torch.randn_like(original_embeddings) * sigma_value
                    perturbed_embeddings = original_embeddings.clone()
                
                # Apply noise only at the selected random index
                    perturbed_embeddings[0, random_token_idx, :] += noise[0, random_token_idx, :]
                
                # 4. Calculate the Log-Likelihood Ratio (LLR or y')
                
                # 4a. Calculate LL for the Predicted Emotion (o)
                # FIX: Corrected method call
                    ll_predicted = self.calculate_ll_from_embeddings(perturbed_embeddings, o_target_ids).item()
                
                # 4b. Calculate LL for the True Emotion (o')
                # FIX: Corrected method call
                    ll_true = self.calculate_ll_from_embeddings(perturbed_embeddings, o_prime_target_ids).item()
                
                # 4c. Calculate LLR (y' = LL(o) - LL(o'))
                # LLR < 1 is used as a filter for successful baseline flip
                    y_prime = ll_predicted - ll_true
                
                # 5. Filter truth samples (y' < 1 indicates a successful flip)
                    if y_prime < 1:
                        truthful_y_primes.append(y_prime)

                results_list.append({
                    'prompt_text': prompt_to_analyze,
                    'true_emotion': true_emotion,
                    'predicted_emotion': predicted_emotion,
                    'y ratio': llr,
                    'noise_token_index': random_token_idx,
                    'noise_token': noise_token,
                    'num_truth_inducing_samples': len(truthful_y_primes),
                    'truthful_y_primes': truthful_y_primes
                })
            
            # Clean up VRAM
            # NOTE: This cleanup logic only works if VRAM usage is significant and should be inside the loop for per-prompt cleanup.
            # However, for a quick cleanup after the main loop, it's fine.
            del original_embeddings
            torch.cuda.empty_cache()

        return pd.DataFrame(results_list)
