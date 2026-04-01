import pytest
import pandas as pd
import torch
from unittest.mock import MagicMock, patch

from src.emotional_mi_pipeline import EmotionAnalysisPipeline

def test_get_rank():
    """
    Tests the core tensor ranking logic in the pipeline.
    Ensures that the 1-based rank is correctly calculated from logits.
    """
    # Create a dummy pipeline without initializing the model
    with patch.object(EmotionAnalysisPipeline, '_load_model'), \
         patch.object(EmotionAnalysisPipeline, '_setup_emotion_tokens', return_value=([], {}, torch.tensor([]))):
        pipeline = EmotionAnalysisPipeline(model_name="dummy", hf_token="dummy", device="cpu")
        
        # Logits sorted descending: index 2 (10.0), index 0 (5.0), index 1 (1.0), index 3 (-2.0)
        logits = torch.tensor([5.0, 1.0, 10.0, -2.0])
        
        # Rank of index 2 should be 1 (highest logit)
        assert pipeline.get_rank(logits, [2]) == 1
        
        # Rank of index 0 should be 2
        assert pipeline.get_rank(logits, [0]) == 2
        
        # Rank of index 3 should be 4
        assert pipeline.get_rank(logits, [3]) == 4
        
        # Rank among multiple tokens should return the minimum rank (best)
        # index 0 is rank 2, index 3 is rank 4 -> best is 2
        assert pipeline.get_rank(logits, [0, 3]) == 2

def test_categorize_prompts():
    """
    Tests the thresholding and categorization of prompts into Extraction vs Enrichment.
    """
    with patch.object(EmotionAnalysisPipeline, '_load_model'), \
         patch.object(EmotionAnalysisPipeline, '_setup_emotion_tokens', return_value=([], {}, torch.tensor([]))):
        pipeline = EmotionAnalysisPipeline(model_name="dummy", hf_token="dummy", device="cpu")
        
        df = pd.DataFrame({
            'prompt': ['p1', 'p2', 'p3'],
            'min_true_rank': [1, 3, 5]
        })
        
        # Median of [1, 3, 5] is 3
        # <= 3 is Extraction, > 3 is Enrichment
        result_df = pipeline.categorize_prompts(df)
        
        assert 'threshold' in result_df.columns
        assert 'hallucination type' in result_df.columns
        assert all(result_df['threshold'] == 3.0)
        
        expected_types = ['Extraction', 'Extraction', 'Enrichment']
        assert result_df['hallucination type'].tolist() == expected_types
