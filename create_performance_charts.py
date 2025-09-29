#!/usr/bin/env python3
"""
Create Model Performance Charts for GitHub Showcase
"""

import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient
import numpy as np
import os

def create_performance_charts():
    """Create performance visualization charts"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    print("🎨 Creating Model Performance Charts...")
    
    # Create images directory
    os.makedirs("images", exist_ok=True)
    
    # Collect all run data
    all_runs_data = []
    experiments = client.search_experiments()
    
    for exp in experiments:
        runs = client.search_runs(exp.experiment_id)
        for run in runs:
            run_data = {
                'experiment': exp.name,
                'run_id': run.info.run_id[:8],
                'model_type': run.data.params.get('model_type', 'Unknown'),
                'test_r2': run.data.metrics.get('test_r2', 0),
                'train_r2': run.data.metrics.get('train_r2', 0),
                'test_mse': run.data.metrics.get('test_mse', 0),
                'train_mse': run.data.metrics.get('train_mse', 0),
                'test_mae': run.data.metrics.get('test_mae', 0),
                'train_mae': run.data.metrics.get('train_mae', 0),
                'status': run.info.status
            }
            all_runs_data.append(run_data)
    
    df = pd.DataFrame(all_runs_data)
    
    if df.empty:
        print("❌ No run data found")
        return
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Chart 1: Model Performance Comparison (R² Scores)
    plt.figure(figsize=(12, 8))
    
    # Create subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🚀 MLflow Model Performance Dashboard', fontsize=16, fontweight='bold')
    
    # 1. R² Score Comparison
    df_sorted = df.sort_values('test_r2', ascending=True)
    bars = ax1.barh(range(len(df_sorted)), df_sorted['test_r2'], 
                    color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'])
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels([f"{row['model_type']}\n({row['run_id']})" for _, row in df_sorted.iterrows()])
    ax1.set_xlabel('R² Score')
    ax1.set_title('📊 Model Accuracy Comparison (R² Score)', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontweight='bold')
    
    # 2. Train vs Test Performance
    ax2.scatter(df['train_r2'], df['test_r2'], s=100, alpha=0.7, 
                c=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'][:len(df)])
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Fit')
    ax2.set_xlabel('Train R² Score')
    ax2.set_ylabel('Test R² Score')
    ax2.set_title('🎯 Train vs Test Performance', fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    # Add model type labels
    for i, row in df.iterrows():
        ax2.annotate(row['model_type'], (row['train_r2'], row['test_r2']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 3. MSE Comparison
    x_pos = np.arange(len(df))
    width = 0.35
    ax3.bar(x_pos - width/2, df['train_mse'], width, label='Train MSE', alpha=0.8, color='#ff6b6b')
    ax3.bar(x_pos + width/2, df['test_mse'], width, label='Test MSE', alpha=0.8, color='#4ecdc4')
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Mean Squared Error')
    ax3.set_title('📉 MSE Comparison (Lower is Better)', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{row['model_type'][:8]}\n({row['run_id']})" for _, row in df.iterrows()], 
                       rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Performance Summary Table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    summary_data = []
    for _, row in df.iterrows():
        summary_data.append([
            row['model_type'],
            f"{row['test_r2']:.4f}",
            f"{row['test_mse']:.4f}",
            f"{row['test_mae']:.4f}",
            row['status']
        ])
    
    table = ax4.table(cellText=summary_data,
                     colLabels=['Model Type', 'R² Score', 'MSE', 'MAE', 'Status'],
                     cellLoc='center',
                     loc='center',
                     colColours=['#f0f0f0']*5)
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title('📋 Performance Summary Table', fontweight='bold', pad=20)
    
    # Highlight best performing row
    best_idx = df['test_r2'].idxmax()
    for j in range(5):
        table[(best_idx + 1, j)].set_facecolor('#90EE90')  # Light green
    
    plt.tight_layout()
    plt.savefig('images/model-performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 2: Experiment Timeline
    plt.figure(figsize=(12, 6))
    
    # Group by experiment
    exp_colors = {'stock_price_prediction': '#ff6b6b', 
                  'stock_price_hyperparameter_tuning': '#4ecdc4'}
    
    for exp_name in df['experiment'].unique():
        exp_data = df[df['experiment'] == exp_name]
        plt.scatter(range(len(exp_data)), exp_data['test_r2'], 
                   label=exp_name, s=100, alpha=0.7, 
                   color=exp_colors.get(exp_name, '#45b7d1'))
        
        # Add model type labels
        for i, (_, row) in enumerate(exp_data.iterrows()):
            plt.annotate(f"{row['model_type']}\n({row['run_id']})", 
                        (i, row['test_r2']), 
                        xytext=(0, 10), textcoords='offset points', 
                        ha='center', fontsize=8)
    
    plt.xlabel('Run Number')
    plt.ylabel('R² Score')
    plt.title('🧪 Experiment Progress & Results', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/experiment-timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 3: Best Model Highlight
    plt.figure(figsize=(10, 6))
    
    best_model = df.loc[df['test_r2'].idxmax()]
    
    metrics = ['R² Score', 'MSE', 'MAE']
    train_values = [best_model['train_r2'], best_model['train_mse'], best_model['train_mae']]
    test_values = [best_model['test_r2'], best_model['test_mse'], best_model['test_mae']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, train_values, width, label='Train', alpha=0.8, color='#ff6b6b')
    plt.bar(x + width/2, test_values, width, label='Test', alpha=0.8, color='#4ecdc4')
    
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title(f'🏆 Best Model Performance: {best_model["model_type"]} (Run: {best_model["run_id"]})', 
              fontsize=14, fontweight='bold')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (train_val, test_val) in enumerate(zip(train_values, test_values)):
        plt.text(i - width/2, train_val + 0.01, f'{train_val:.4f}', 
                ha='center', va='bottom', fontweight='bold')
        plt.text(i + width/2, test_val + 0.01, f'{test_val:.4f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('images/best-model-performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Performance charts created successfully!")
    print("📁 Files saved:")
    print("   - images/model-performance.png (Main dashboard)")
    print("   - images/experiment-timeline.png (Progress over time)")
    print("   - images/best-model-performance.png (Best model details)")
    
    # Print summary
    print(f"\n📊 Performance Summary:")
    print(f"   Total Models: {len(df)}")
    print(f"   Best R² Score: {df['test_r2'].max():.4f}")
    print(f"   Best Model: {best_model['model_type']}")
    print(f"   Experiments: {df['experiment'].nunique()}")

if __name__ == "__main__":
    create_performance_charts()