import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from typing import List, Tuple, Optional, List
from emotional_mi_pipeline import EmotionAnalysisPipeline
from .attention_weights_analysis import AttentionWeights

class MiVisualisations:

    def __init__(self, mi_pipeline):
        """
        Initializes the visualization class with an analysis pipeline.
        
        Args:
            mi_pipeline: An instance of a pipeline class (e.g., EmotionAnalysisPipeline)
                         that contains the model, tokenizer, and analysis methods.
        """
        self.mi_pipeline = mi_pipeline
        self.model = mi_pipeline.model
        self.tokenizer = mi_pipeline.tokenizer
        self.device = mi_pipeline.device
        self.attn_analyzer = mi_pipeline
        

    def visualize_single_prompt_attention(self, prompt: str, layer: int, head: int, figsize: Tuple[int, int], save_filepath: Optional[str] = None):
        """
        Calculates and plots the TokenLength x TokenLength attention heatmap 
        for a single prompt, layer, and head, labeling axes with input tokens.
        
        Args:
            prompt (str): The specific input text prompt to analyze.
            layer (int): The layer index (0 to N-1) to extract attention from.
            head (int): The head index (0 to N-1) within the specified layer.
            figsize (tuple): The size of the matplotlib figure (width, height).
            save_filepath (Optional[str]): Path to save the plot (e.g., 'heatmap.png').
        """
        # raw attention matrix (T x T) using the analysis helper class
        attention_to_plot: Optional[np.ndarray] = self.attn_analyzer.analyze_single_prompt_attn_wts(prompt, layer, head)
        
        if attention_to_plot is None:
            print("Visualization aborted due to error in attention extraction.")
            return

        
        try:
            
            tokenizer = self.attn_analyzer.tokenizer
            # Tokenize the prompt and include the BOS token, consistent with model input
            input_tokens = tokenizer.tokenize(prompt)
            input_tokens = [tokenizer.bos_token] + input_tokens
        except AttributeError:
            print("Error: Tokenizer not found on the analysis object.")
            seq_len = attention_to_plot.shape[0]
            input_tokens = [f"T{i}" for i in range(seq_len)]
        except Exception as e:
            print(f"Warning: Could not tokenize prompt for axis labels: {e}. Using indices.")
            seq_len = attention_to_plot.shape[0]
            input_tokens = [f"T{i}" for i in range(seq_len)]


        #  Plotting the heatmap
        print(f"\nVisualizing Attention Heatmap for Layer {layer}, Head {head} ---")
            
        plt.figure(figsize=figsize)
        sns.heatmap(attention_to_plot, 
                    xticklabels=input_tokens, 
                    yticklabels=input_tokens,
                    cmap="viridis",
                    linewidths=0.5,
                    linecolor='gray')
        
        plt.title(f"Raw Attention Heatmap: Layer {layer}, Head {head} (Prompt: '{prompt[:30]}...')", fontsize=10)
        plt.xlabel("Key Tokens (Token being attended TO)", fontsize=12)
        plt.ylabel("Query Tokens (Token doing the attending FROM)", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0) 
        plt.tight_layout()
        
        if save_filepath:
            plt.savefig(save_filepath, bbox_inches='tight')
            print(f"Plot saved to: {save_filepath}")
            
        plt.show()
        plt.close()

        
    @staticmethod    
    def plot_attention_heatmap(attention_avg: np.ndarray, title: str, figsize: Tuple[int, int], save_filepath: Optional[str] = None) -> None:
        """
        Generates a standard heatmap for a single averaged attention matrix (Layer x Head).

        Args:
            attention_avg (np.ndarray): Average attention values for a single sample (Heads x Layers).
            title (str): Title for the plot.
            figsize (Tuple[int, int]): Size of the figure.
            save_filepath (Optional[str]): Path to save the plot (e.g., 'heatmap.png').
        """
        if attention_avg.size == 0:
            print(f"Skipping plot: {title} - empty data.")
            return

        num_heads, num_layers = attention_avg.shape # Assuming shape is (Heads, Layers)
    
        v_max = attention_avg.max() if attention_avg.size > 0 else 0.1

        plt.figure(figsize=figsize)
    
        sns.heatmap(
            attention_avg,
            cmap='viridis', 
            vmax=v_max,
            cbar_kws={'label': 'Average Attention Contribution'},
            xticklabels=np.arange(num_layers),
            yticklabels=np.arange(num_heads)
        )


        plt.title(f'{title}\nAverage Attention Head Activity', fontsize=12, fontweight='bold')
        plt.xlabel('Layers', fontsize=12)
        plt.ylabel('Heads', fontsize=12)
        plt.tight_layout()
        
        if save_filepath:
            plt.savefig(save_filepath, bbox_inches='tight')
            print(f"Plot saved to: {save_filepath}")
            
        plt.show() 
        plt.close()
        
        
    
    @staticmethod
    def plot_attention_sample_heatmaps(matrices: List[np.ndarray], labels: List[str], n_cols: Optional[int] = None,title: str = "Attention Contribution (Heads vs. Layers)",is_differential: bool = False, save_filepath: Optional[str] = None) -> None:    
        """
        Generates a grid of heatmaps for a list of attention matrices, ensuring a 
        consistent color scale across all plots for comparative analysis. Automatically
        uses a diverging color scale if is_differential is True.

        Args:
            matrices (List[np.ndarray]): A list of NumPy arrays (Heads x Layers).
            labels (List[str]): A list of titles for each subplot.
            n_cols (Optional[int]): The number of columns in the subplot grid. 
            title (str): The overall suptitle for the entire figure.
            is_differential (bool): If True, plots with a symmetric, diverging color
                                 scale (RdBu_r) centered at zero.
            save_filepath (Optional[str]): Path to save the plot (e.g., 'heatmaps_grid.png').
        """
        n_plots = len(matrices)
    
        if n_plots == 0:
            print("Error: No matrices were provided to plot.")
            return

    
        if n_plots != len(labels):
            raise ValueError(
                f"Length mismatch. The number of labels ({len(labels)}) must exactly match the number of matrices ({n_plots})."
            )

    
        if n_cols is None:
            if n_plots <= 5:
                n_cols = n_plots
            elif n_plots == 6:
                n_cols = 3
            else:
                n_cols = 5
            
        n_rows = math.ceil(n_plots / n_cols)
    
        
        # Determine global max/min for shared color scaling
        if is_differential:
            v_abs_max = max(abs(m).max() for m in matrices)
            vmin = -v_abs_max
            vmax = v_abs_max
            cmap = "RdBu_r"  # Diverging map for differences
            cbar_label = 'Differential Contribution (Run A - Run B)'
        else:
            vmin = 0  
            vmax = max(m.max() for m in matrices)
            cmap = "viridis" # Sequential map for raw scores
            cbar_label = 'Attention Contribution'
    
        # Set figure size based on the grid structure
        figsize_w = n_cols * 6
        figsize_h = n_rows * 5
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_w, figsize_h), squeeze=False)
        axes = axes.flatten()

        print(f"Plotting {n_plots} maps in a {n_rows}x{n_cols} grid.")

        for i in range(n_plots):
            ax = axes[i]
            matrix = matrices[i]
            label = labels[i]
        
            sns.heatmap(
                matrix,
                ax=ax,
                cmap=cmap, 
                cbar=True,
                vmin=vmin, 
                vmax=vmax, 
                cbar_kws={'label': cbar_label} 
            )
            
            ax.set_title(f"{i+1}. {label}", fontsize=14)
            num_heads, num_layers = matrix.shape
            # Y-axis (Heads) labels
            if i % n_cols == 0:
                ax.set_ylabel("Heads", fontsize=12)
                ax.set_yticks(np.arange(0, num_heads, 4) + 0.5, labels=np.arange(0, num_heads, 4), rotation=0)
            else:
                ax.set_yticks([]) 
            # X-axis (Layers) labels
            if i >= (n_plots - n_cols) or n_rows == 1:
                ax.set_xlabel("Layers", fontsize=12)
                ax.set_xticks(np.arange(0, num_layers, 4) + 0.5, labels=np.arange(0, num_layers, 4))
            else:
                ax.set_xticks([]) 
    
        for j in range(n_plots, n_rows * n_cols):
            fig.delaxes(axes[j])
    
        fig.suptitle(title, fontsize=18, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        if save_filepath:
            plt.savefig(save_filepath, bbox_inches='tight')
            print(f"Plot saved to: {save_filepath}")
            
        plt.show()
        plt.close()

    
    @staticmethod
    def plot_logit_lens_trace_single(df: pd.DataFrame, title: str, figsize: Tuple[int, int] = (14, 8), plot_difference: bool = False, save_filepath: Optional[str] = None) -> None:
        
        """
        Generates a standard Logit Lens Plot showing the change in raw logit scores for 
        true and predicted tokens across transformer layers, averaged across prompts.
    
        This version plots the traces for a single set of true/predicted emotions.

        Args:
            df (pd.DataFrame): The DataFrame containing averaged logit metrics.
            title (str): The plot title.
            figsize (Tuple[int, int]): Size of the figure.
            plot_difference (bool): If True, plots the logit_difference_mean and its STD 
                                 instead of the raw scores.
            save_filepath (Optional[str]): Path to save the plot (e.g., 'single_trace.png').
        """
        if df.empty:
            print("DataFrame for plotting is empty.")
            return
        
        plt.figure(figsize=figsize)
    
        max_layer = df['layer'].max()
    
        # Plotting True and Predicted Raw Logit Scores
        
        if not plot_difference:
            true_color = 'darkcyan'
            predicted_color = 'firebrick'

            # True Logit Score Plot
            plt.plot(
                df['layer'],
                df['true_logit_raw_mean'],
                marker='o',
                markersize=6,
                linestyle='-',
                linewidth=2,
                color=true_color,
                label=f"True Logit Score ({df['true_emotion'].iloc[0]})"
            )

            plt.fill_between(
                df['layer'],
                df['true_logit_raw_mean'] - df['true_logit_std'],
                df['true_logit_raw_mean'] + df['true_logit_std'],
                color=true_color,
                alpha=0.10,
            )

            # Predicted Logit Score Plot
            plt.plot(
                df['layer'],
                df['predicted_logit_raw_mean'],
                marker='o',
                markersize=6,
                linestyle='--',
                linewidth=2,
                color=predicted_color,
                label=f"Predicted Logit Score ({df['predicted_emotion'].iloc[0]})"
            )

            plt.fill_between(
                df['layer'],
                df['predicted_logit_raw_mean'] - df['predicted_logit_std'],
                df['predicted_logit_raw_mean'] + df['predicted_logit_std'],
                color=predicted_color,
                alpha=0.10,
            )
    
        # Conditional Plotting of Logit Difference
        
        if plot_difference:
            difference_color = 'purple'
        
            # Plot mean difference (Predicted - True)
            plt.plot(
                df['layer'],
                df['logit_difference_mean'],
                marker='o',
                markersize=7,
                linestyle='-',
                linewidth=2,
                color=difference_color,
                label=f"Logit Difference (Pred - True)"
            )
        
            # Plot difference standard deviation
            plt.fill_between(
                df['layer'],
                df['logit_difference_mean'] - df['logit_difference_std'],
                df['logit_difference_mean'] + df['logit_difference_std'],
                color=difference_color,
                alpha=0.15,
            )
    
            plt.ylabel('Logit Difference', fontsize=14)
        else:
            plt.ylabel('Average Logit Score', fontsize=14)

        plt.title(f'{title}', fontsize=18, fontweight='bold')
        plt.xlabel('Transformer Layers', fontsize=14)
        plt.xlim(-1, max_layer + 1)
    
    
        if max_layer > 0:
            plt.xticks(np.arange(0, max_layer + 1, 2))
        
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left', fontsize=10)
        plt.tight_layout()
        
        if save_filepath:
            plt.savefig(save_filepath, bbox_inches='tight')
            print(f"Plot saved to: {save_filepath}")
            
        plt.show()
        plt.close()

        

    @staticmethod
    def plot_logit_lens_trace_multi(data_list: List[Tuple[pd.DataFrame, str]], title: str, figsize: Tuple[int, int], save_filepath: Optional[str] = None) -> None:
        """
        Generates a generalized Logit Lens Plot showing the change in raw logit scores for 
        true and predicted tokens across transformer layers, averaged across prompts, 
        allowing comparison across multiple datasets/experiments.
    
        The input data_list is a list of tuples: [(DataFrame, Label_for_Legend)].
        
        Each DataFrame MUST contain these columns: 
        'layer', 'true_logit_raw_mean', 'predicted_logit_raw_mean', 
        'true_logit_std', and 'predicted_logit_std'.
        
        Args:
            data_list (List[Tuple[pd.DataFrame, str]]): List of data and labels for plotting.
            title (str): The plot title.
            figsize (Tuple[int, int]): Size of the figure.
            save_filepath (Optional[str]): Path to save the plot (e.g., 'multi_trace.png').
        """
        
        if not data_list:
            print("Data list for plotting is empty.")
            return

        plt.figure(figsize=figsize)
        base_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown'] 
    
        max_layer = 0
    
        for i, (df, label) in enumerate(data_list):
            if df.empty:
                print(f"Skipping empty DataFrame for label: {label}")
                continue

            color_index = i % len(base_colors)
            line_color = base_colors[color_index]
            max_layer = max(max_layer, df['layer'].max())
        
            # Plot True Logit Score Trace
            plt.plot(
                df['layer'],
                df['true_logit_raw_mean'], 
                markersize=6,
                linestyle='-',
                linewidth=2,
                color=line_color,
                label=f"{label} - True Score"
            )
        
            # Plot True Logit Score STD area
            plt.fill_between(
                df['layer'],
                df['true_logit_raw_mean'] - df['true_logit_std'], 
                df['true_logit_raw_mean'] + df['true_logit_std'], 
                color=line_color,
                alpha=0.15, 
            )
        
        
            """
            
            plt.plot(
                df['layer'],
                df['predicted_logit_raw_mean'],
                linestyle='--',
                linewidth=2,
                color=line_color,
                label=f"{label} - Predicted Score"
            )
            plt.fill_between(
                df['layer'],
                df['predicted_logit_raw_mean'] - df['predicted_logit_std'],
                df['predicted_logit_raw_mean'] + df['predicted_logit_std'],
                color=line_color,
                alpha=0.08,
            )
            """

    
        plt.title(f'{title}', fontsize=12, fontweight='bold')
        plt.xlabel('Transformer Layers', fontsize=14)
        plt.ylabel('Average Logit Score', fontsize=14)
        plt.xlim(-1, max_layer + 1) 
    
        if max_layer > 0:
            plt.xticks(np.arange(0, max_layer + 1, max(1, max_layer // 16)))
            
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left', fontsize=8)
        plt.tight_layout()
        
        if save_filepath:
            plt.savefig(save_filepath, bbox_inches='tight')
            print(f"Plot saved to: {save_filepath}")
            
        plt.show()
        plt.close()
    
    
    @staticmethod
    def plot_overlay_traces(dfs_list: List[pd.DataFrame], title: str, figsize: Tuple[int, int], metric_column: str = 'log_likelihood_ratio', save_filepath: Optional[str] = None) -> None:
        """
        Combines a list of DataFrames (each representing a single prompt's trace) 
        into one long DataFrame, calculates the mean trace, and plots all individual traces 
        overlaid with the overall mean.
    
        Args:
            dfs_list (List[pd.DataFrame]): List of individual trace DataFrames. Each DF 
                                         must contain the column specified by metric_column 
                                         and 'layer'.
            title (str): The plot title.
            figsize (Tuple[int, int]): Size of the figure.
            metric_column (str): The column name to plot and average. Must be one of
                                 'log_likelihood_ratio' or 'logit_difference_mean'.
            save_filepath (Optional[str]): Path to save the plot (e.g., 'overlay_traces.png').
        """
    
        valid_metrics = ['log_likelihood_ratio', 'logit_difference_mean']
        
        if metric_column not in valid_metrics:
            print(f"Error: metric_column must be one of {valid_metrics}")
            return
        
        if not dfs_list:
            print("Error: The input list of DataFrames is empty.")
            return

        n_samples = len(dfs_list)
    
        # Concatenate all individual traces
        df_all_traces = pd.concat(
            dfs_list, 
            keys=[f'Trace_{i+1}' for i in range(n_samples)], 
            names=['sample_id', 'index']
        ).reset_index()

        # Calculate the mean trace for the selected metric
        df_avg = df_all_traces.groupby('layer')[metric_column].mean().reset_index()

    
        plt.figure(figsize=figsize)

        unique_samples = df_all_traces['sample_id'].unique()
    
        # Individual Traces
        for i, sample_id in enumerate(unique_samples):
            sample_data = df_all_traces[df_all_traces['sample_id'] == sample_id]
        
            # Only label the first trace to avoid clutter in the legend
            label = f'Individual Sample Trace (N={n_samples})' if i == 0 else "_nolegend_"
        
            plt.plot(
                sample_data['layer'],
                sample_data[metric_column], # Use selected metric
                color='lightsteelblue', 
                alpha=0.2, 
                linewidth=1,
                label=label
            )

        # Plot Overall Mean Trace 
        if metric_column == 'log_likelihood_ratio':
            y_label = 'Log-Likelihood Ratio (LLR)'
        else: # logit_difference_mean
            y_label = 'Logit Difference (Predicted - True)'
    
        plt.plot(
            df_avg['layer'],
            df_avg[metric_column],
            color='red',
            linestyle='-',
            linewidth=4,
            label=f'Overall Mean Trace (N={n_samples})',
            zorder=5 
        )

        plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Decision Boundary (Metric = 0)')
    
        plt.title(f'{title}', fontsize=14, fontweight='bold')
        plt.xlabel('Transformer Layer', fontsize=14)
        plt.ylabel(y_label, fontsize=14)
    
        max_layer = df_avg['layer'].max()
        plt.xlim(-1, max_layer + 1)
            
        if max_layer > 0:
            plt.xticks(np.arange(0, max_layer + 1, 2))
        
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.legend(loc='upper left', fontsize=12)
        plt.tight_layout()
        
        if save_filepath:
            plt.savefig(save_filepath, bbox_inches='tight')
            print(f"Plot saved to: {save_filepath}")
            
        plt.show()
        plt.close()
        
        
    @staticmethod
    def plot_final_layer_distribution(dfs_list: List[pd.DataFrame],title: str,figsize: Tuple[int, int] = (10, 8), save_filepath: Optional[str] = None) -> None:
        """
        Combines a list of DataFrames (each assumed to be an already-averaged trace 
        from a single run/batch), filters to the final layer, and plots the distribution 
        of the final layer's mean metric.
    
        Dynamically supports both 'log_likelihood_ratio' and 'logit_difference_mean' columns.

        Args:
            dfs_list (List[pd.DataFrame]): List of averaged trace DataFrames (e.g., df_avg_final_LL_metrics1).
            title (str): The plot title.
            figsize (Tuple[int, int]): Size of the figure.
            save_filepath (Optional[str]): Path to save the plot (e.g., 'distribution.png').
        """
        if not dfs_list:
            print("Error: The input list of DataFrames is empty.")
            return
        
        df_sample = dfs_list[0]
        
        ll_options = ['log_likelihood_ratio', 'log_likelihood_ratio_mean']
        logit_diff_options = ['logit_difference_mean', 'logit_difference']
        
        metric_col = None
        y_label = ''

        if any(col in df_sample.columns for col in ll_options):
            metric_col = next(col for col in ll_options if col in df_sample.columns)
            y_label = 'Final Layer Log-Likelihood Ratio (LLR)'
        elif any(col in df_sample.columns for col in logit_diff_options):
            metric_col = next(col for col in logit_diff_options if col in df_sample.columns)
            y_label = 'Final Layer Logit Difference (Predicted - True)'
        else:
            print("Error: Input DataFrames must contain a valid metric column.")
            return


        df_all_traces = pd.concat(dfs_list, ignore_index=True)

        final_layer = df_all_traces['layer'].max()
        df_final_layer = df_all_traces[df_all_traces['layer'] == final_layer].copy()
    
        df_final_layer['metric_value'] = df_final_layer[metric_col]

        mean_diff = df_final_layer['metric_value'].mean()
        median_diff = df_final_layer['metric_value'].median()
        n_samples = len(df_final_layer)

   
        fig, ax2 = plt.subplots(figsize=figsize)
        
        # box plot for summary statistics
        sns.boxplot(
            y=df_final_layer['metric_value'],
            color='skyblue',
            width=0.4,
            linewidth=1.5,
            ax=ax2,
            boxprops=dict(edgecolor='black'),
            medianprops=dict(color='black', linewidth=2)
        )

        # 7. Overlay strip plot
        sns.stripplot(
            y=df_final_layer['metric_value'],
            color='forestgreen',
            alpha=0.7,
            size=7,
            jitter=0.2,
            edgecolor='black',
            linewidth=0.5,
            ax=ax2
        )

      
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)

    
        ax2.text(
            0.55, median_diff, f'Median: {median_diff:.2f}', 
            ha='center', va='bottom', color='black',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.3'), 
            transform=ax2.get_xaxis_transform()
        )
        
        ax2.text(
            0.45, mean_diff, f'Mean: {mean_diff:.2f}', 
            ha='center', va='top', color='darkred', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkred', boxstyle='round,pad=0.3'), 
            transform=ax2.get_xaxis_transform()
        )

        ax2.set_title(
            f'{title}\nDistribution of Final Layer Mean Metric (Layer {final_layer}) (N={n_samples} Runs)', 
            fontsize=16, 
            fontweight='bold'
        )
        ax2.set_ylabel(y_label, fontsize=14)
        ax2.set_xlabel('')
        ax2.set_xticks([]) 
        ax2.grid(axis='y', linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        
        if save_filepath:
            plt.savefig(save_filepath, bbox_inches='tight')
            print(f"Plot saved to: {save_filepath}")
            
        plt.show()
        plt.close()
