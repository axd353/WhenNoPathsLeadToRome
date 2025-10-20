import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
def plot_distribution_with_percentile(data, column, title):
    """Plots the distribution of a column and prints max and 90th percentile."""
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], kde=True)
    max_val = data[column].max()
    percentile_90 = data[column].quantile(0.9)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.text(0.95, 0.95, f"Max: {max_val:.2f}", transform=plt.gca().transAxes,
             ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.95, 0.88, f"90th: {percentile_90:.2f}", transform=plt.gca().transAxes,
             ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    plt.show()

def plot_multi_distributions_with_percentiles(data, columns, title_prefix="Distribution of"):
    """
    Plots distributions of multiple columns in a grid layout with max and 90th percentile values.
    
    Parameters:
    - data: DataFrame containing the data
    - columns: List of column names to plot
    - title_prefix: Prefix for each subplot title (default: "Distribution of")
    """
    n_cols = len(columns)
    
    # Determine grid layout
    if n_cols <= 3:
        n_rows = 1
        n_cols = n_cols
    else:
        n_rows = ceil(n_cols / 3)
        n_cols = 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle(f"{title_prefix} Features", y=1.02, fontsize=14)
    
    # Flatten axes array for easy iteration
    if n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if n_cols == 1 else axes
    
    for i, (col, ax) in enumerate(zip(columns, axes)):
        sns.histplot(data[col], kde=True, ax=ax)
        max_val = data[col].max()
        percentile_90 = data[col].quantile(0.9)
        
        ax.set_title(f"{title_prefix} {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        
        ax.text(0.95, 0.95, f"Max: {max_val:.2f}", transform=ax.transAxes,
                ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
        ax.text(0.95, 0.88, f"90th: {percentile_90:.2f}", transform=ax.transAxes,
                ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
    
    # Hide any unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()