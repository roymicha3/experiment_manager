# Training Curves Visualization Guide

This guide shows you how to create training curves graphs using the experiment_manager visualization system.

## 🚀 Quick Start

### Basic Training Curves (Train/Validation)

```python
import pandas as pd
from experiment_manager.visualization.plots.training_curves import TrainingCurvesPlotPlugin
from experiment_manager.visualization.plugins.plot_plugin import PlotData

# 1. Prepare your data in this format
data = [
    {'step': 1, 'metric_name': 'loss', 'value': 2.3, 'split': 'train'},
    {'step': 1, 'metric_name': 'loss', 'value': 2.4, 'split': 'val'},
    {'step': 2, 'metric_name': 'loss', 'value': 1.8, 'split': 'train'},
    {'step': 2, 'metric_name': 'loss', 'value': 1.9, 'split': 'val'},
    # ... more data points
]

df = pd.DataFrame(data)

# 2. Create the plot
plugin = TrainingCurvesPlotPlugin()
plugin.initialize()

config = {
    'figsize': (12, 6),
    'title': 'Training and Validation Loss',
    'smoothing': 'moving_average',
    'smoothing_params': {'window': 5},
    'xlabel': 'Epoch',
    'grid': True
}

result = plugin.generate_plot(PlotData(df), config)

# 3. Save the plot
if result.success:
    fig = result.plot_object
    fig.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
```

## 📊 Data Format Requirements

### Required Columns
- `step`: Epoch or batch number (numeric)
- `metric_name`: Name of the metric (e.g., 'loss', 'accuracy')
- `value`: The metric value (numeric)

### Optional Columns
- `split`: Train/validation split ('train', 'val', 'validation', 'test')
- `run_id`: For multiple runs with confidence bands
- `experiment_id`: Alternative to run_id

### Example Data Structure
```python
# Single run with train/val splits
{
    'step': [1, 1, 2, 2, 3, 3],
    'metric_name': ['loss', 'loss', 'loss', 'loss', 'loss', 'loss'],
    'value': [2.3, 2.4, 1.8, 1.9, 1.4, 1.6],
    'split': ['train', 'val', 'train', 'val', 'train', 'val']
}

# Multiple runs
{
    'step': [1, 1, 1, 1],
    'metric_name': ['loss', 'loss', 'loss', 'loss'],
    'value': [2.3, 2.4, 2.2, 2.5],
    'split': ['train', 'val', 'train', 'val'],
    'run_id': [0, 0, 1, 1]
}
```

## ⚙️ Configuration Options

### Basic Configuration
```python
config = {
    'figsize': (12, 6),          # Figure size (width, height)
    'title': 'My Training Curves', # Plot title
    'xlabel': 'Epoch',           # X-axis label
    'grid': True,                # Show grid
    'linewidth': 2               # Line thickness
}
```

### Smoothing Options
```python
# Moving average smoothing
'smoothing': 'moving_average',
'smoothing_params': {'window': 5}

# Gaussian smoothing
'smoothing': 'gaussian',
'smoothing_params': {'sigma': 1.5}

# Exponential smoothing
'smoothing': 'exponential',
'smoothing_params': {'alpha': 0.3}

# No smoothing
'smoothing': 'none'
```

### Color Customization
```python
config = {
    'split_colors': {
        'train': '#2E86C1',      # Blue for training
        'val': '#E74C3C',        # Red for validation
        'test': '#28B463'        # Green for test
    },
    'base_color': 'blue',        # Default color
    'mean_color': 'darkblue',    # Mean line color
    'confidence_color': 'blue'   # Confidence band color
}
```

### Multiple Runs & Confidence Bands
```python
config = {
    'show_confidence': True,     # Show confidence bands
    'confidence_level': 0.95,    # 95% confidence interval
    'confidence_alpha': 0.2,     # Confidence band transparency
    'individual_alpha': 0.3,     # Individual run transparency
    'mean_linewidth': 3          # Mean line thickness
}
```

## 🎯 Use Cases & Examples

### 1. Basic Train/Val Curves
Perfect for monitoring single training runs.

### 2. Multiple Metrics
Plot loss and accuracy side by side in subplots.

### 3. Multiple Runs with Confidence
Compare training stability across multiple runs.

### 4. Learning Rate Schedule Analysis
Add annotations to mark LR changes.

### 5. Hyperparameter Comparison
Compare different experimental setups.

## 🔧 Advanced Features

### Annotations
```python
annotations = [
    {
        'type': 'vertical_line',
        'x': 20,                 # Epoch 20
        'color': 'orange',
        'linestyle': '--',
        'label': 'LR Drop'
    },
    {
        'type': 'text',
        'x': 10, 'y': 1.5,
        'text': 'Initial Phase',
        'fontsize': 10
    }
]

config['annotations'] = annotations
```

### Custom Axis Limits
```python
config = {
    'xlim': [0, 100],           # X-axis limits
    'ylim': [0, 2.5]            # Y-axis limits
}
```

## 💡 Tips & Best Practices

1. **Data Format**: Always use the long format (one row per data point)
2. **Smoothing**: Use moving average for noisy data, none for clean data
3. **Multiple Runs**: Include run_id for confidence bands
4. **Colors**: Use consistent colors across plots (blue=train, red=val)
5. **File Naming**: Use descriptive names like 'loss_curves_experiment_1.png'

## 🔍 Converting Your Data

### From PyTorch/TensorFlow Logs
```python
# If you have data like this:
logs = {
    'epoch': [1, 2, 3, 4, 5],
    'train_loss': [2.3, 1.8, 1.4, 1.1, 0.9],
    'val_loss': [2.4, 1.9, 1.6, 1.3, 1.1]
}

# Convert to required format:
data = []
for i, epoch in enumerate(logs['epoch']):
    data.extend([
        {'step': epoch, 'metric_name': 'loss', 'value': logs['train_loss'][i], 'split': 'train'},
        {'step': epoch, 'metric_name': 'loss', 'value': logs['val_loss'][i], 'split': 'val'}
    ])

df = pd.DataFrame(data)
```

### From CSV Files
```python
# If you have a CSV with columns: epoch, train_loss, val_loss, train_acc, val_acc
df_raw = pd.read_csv('training_logs.csv')

# Melt to long format
df = df_raw.melt(
    id_vars=['epoch'], 
    value_vars=['train_loss', 'val_loss', 'train_acc', 'val_acc'],
    var_name='metric_split', 
    value_name='value'
)

# Extract metric and split
df[['split', 'metric_name']] = df['metric_split'].str.split('_', n=1, expand=True)
df['step'] = df['epoch']
df = df[['step', 'metric_name', 'value', 'split']]
```

## 🎨 Examples Generated

Run `python training_curves_examples.py` to generate these sample plots:

1. **example_1_basic_training_curves.png**: Simple train/val loss
2. **example_2_multiple_metrics.png**: Loss and accuracy subplots  
3. **example_3_multiple_runs.png**: Multiple runs with confidence bands
4. **example_4_plot_spec.png**: Using PlotSpec system
5. **example_5_with_annotations.png**: Learning rate schedule markers
6. **example_6_real_data.png**: From realistic experiment logs

Each example demonstrates different aspects of the training curves system! 