import pytest
import pandas as pd

from src.analysis.causal_validation_analysis import CausalValidationAnalysis

def test_get_positive_LLR_prompts():
    """
    Tests the filtering of causal tracing results to extract positive LLRs for a target layer.
    """
    data = {
        'prompt_text': ['p1', 'p2', 'p3', 'p4'],
        'true_emotion': ['joy', 'sad', 'anger', 'fear'],
        'predicted_emotion': ['sad', 'joy', 'fear', 'anger'],
        'layer': [30, 31, 31, 31],
        'log_likelihood_ratio': [5.0, -2.0, 3.5, 0.0],
        'extra_col': [1, 2, 3, 4]
    }
    df = pd.DataFrame(data)
    
    filtered_df = CausalValidationAnalysis.get_positive_LLR_prompts(df, target_layer=31)
    
    assert not filtered_df.empty
    assert len(filtered_df) == 1
    assert filtered_df.iloc[0]['prompt_text'] == 'p3'
    
    expected_cols = ['prompt_text', 'true_emotion', 'predicted_emotion', 'layer', 'log_likelihood_ratio']
    assert set(filtered_df.columns) == set(expected_cols)
