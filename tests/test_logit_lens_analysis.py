import pytest
import pandas as pd
import torch
from unittest.mock import MagicMock, patch

# Adjust the import path if needed based on your testing environment setup
from src.emotional_mi_pipeline import EmotionAnalysisPipeline
from src.analysis.logit_lens_analysis import LogitLensAnalysis

@pytest.fixture
def mock_pipeline():
    """Creates a heavily mocked EmotionAnalysisPipeline."""
    pipeline = MagicMock(spec=EmotionAnalysisPipeline)
    
    # Mock model properties
    pipeline.model = MagicMock()
    pipeline.model.cfg = MagicMock()
    pipeline.model.cfg.n_layers = 2
    pipeline.device = "cpu"
    
    # We will assume a hidden dimension of 768 and a vocab size of 32000
    hidden_dim = 768
    vocab_size = 32000
    seq_len = 5
    
    # 1. Mock run_with_cache to return a dummy activation cache
    # The LogitLensAnalysis relies on cache[("mlp_out", layer_idx)]
    # which expects shape [batch_size, seq_len, hidden_dim]
    dummy_cache = {
        ("mlp_out", 0): torch.randn(1, seq_len, hidden_dim),
        ("mlp_out", 1): torch.randn(1, seq_len, hidden_dim),
    }
    pipeline.model.run_with_cache.return_value = (None, dummy_cache)
    
    # 2. Mock model.unembed to perform the LogitLens "projection"
    # It must take a tensor of [hidden_dim] and return [vocab_size]
    def mock_unembed(activation):
        assert activation.shape == (hidden_dim,), f"Expected shape {(hidden_dim,)}, got {activation.shape}"
        # Return a randomly initialized tensor of vocab shape
        return torch.randn(vocab_size)
    
    pipeline.model.unembed.side_effect = mock_unembed
    
    # 3. Mock the tokenizer
    pipeline.tokenizer = MagicMock()
    # Mock encode to return a sequence of `seq_len` tokens
    pipeline.tokenizer.encode.return_value = torch.ones(1, seq_len, dtype=torch.long)
    pipeline.model.tokenizer = pipeline.tokenizer
    
    # 4. Mock token ID and rank helpers
    pipeline.get_token_ids.side_effect = lambda emotion: [123] if emotion == 'joy' else [456]
    pipeline.get_rank.return_value = 1  # Dummy rank
    
    return pipeline, vocab_size

def test_logit_lens_projection_vocab_shape(mock_pipeline):
    """
    Verifies that the LogitLens analysis correctly extracts the MLP activation,
    projects it (via unembed) into the vocabulary space, and processes the logit scores.
    """
    pipeline, vocab_size = mock_pipeline
    
    # Initialize the analyzer with our mock
    analyzer = LogitLensAnalysis(mi_pipeline=pipeline)
    
    # Run the single prompt analysis
    df_result = analyzer.analyze_logit_single_prompt_mlp(
        prompt="The man was feeling very happy.",
        true_emotion="joy",
        predicted_emotion="sad"
    )
    
    # Verify that the pipeline processed exactly n_layers
    assert not df_result.empty
    assert len(df_result) == 2  # n_layers = 2
    
    # Verify unembed was called exactly n_layers times
    assert pipeline.model.unembed.call_count == 2
    
    # Introspect the arguments used to call get_rank to ensure they have the vocabulary shape
    # get_rank is called twice per layer (for true and predicted emotion)
    assert pipeline.get_rank.call_count == 4
    for call_args in pipeline.get_rank.call_args_list:
        logits_tensor, token_ids = call_args[0]
        # This is the core check: does the projected tensor have the [vocab_size] shape?
        assert logits_tensor.shape == (vocab_size,)
        assert isinstance(logits_tensor, torch.Tensor)

    # Verify expected dataframe columns are present
    expected_cols = ['layer', 'true_logit_raw', 'predicted_logit_raw', 'logit_difference', 'true_rank', 'predicted_rank']
    for col in expected_cols:
        assert col in df_result.columns
