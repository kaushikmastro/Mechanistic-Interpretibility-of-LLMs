import pytest
import pandas as pd
import numpy as np

from src.analysis.attention_weights_analysis import AttentionWeights

def test_average_attention_matrix_contributions():
    """
    Tests the static method for converting raw attention metrics into a proper LayerxHead matrix.
    """
    # Create sample raw metrics
    data = [
        {'layer': 0, 'head': 0, 'attention_to_previous_token': 0.1},
        {'layer': 0, 'head': 0, 'attention_to_previous_token': 0.3},
        {'layer': 0, 'head': 1, 'attention_to_previous_token': 0.5},
        {'layer': 1, 'head': 0, 'attention_to_previous_token': 0.8},
        {'layer': 1, 'head': 1, 'attention_to_previous_token': 0.4},
    ]
    df = pd.DataFrame(data)
    
    matrix = AttentionWeights.average_attention_matrix_contributions(df)
    
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 2)
    assert np.isclose(matrix[0, 0], 0.2)
    assert np.isclose(matrix[0, 1], 0.5)
    assert np.isclose(matrix[1, 0], 0.8)
    assert np.isclose(matrix[1, 1], 0.4)

def test_top_attn_heads_contributions():
    """
    Tests the static method for finding top k heads based on differential absolute scores.
    """
    matrix1 = np.array([[ 0.5, -0.2], 
                         [-0.9,  0.1]])
    matrix2 = np.array([[ 0.1,  0.4], 
                         [ 0.3, -0.7]])
    
    top_heads_df = AttentionWeights.top_attn_heads_contributions([matrix1, matrix2], k=2)
    
    assert not top_heads_df.empty
    assert len(top_heads_df) == 2
    assert top_heads_df.iloc[0]['layer'] == 1
    assert top_heads_df.iloc[0]['head'] == 0
    assert np.isclose(top_heads_df.iloc[0]['average_abs_contribution'], 0.6)
