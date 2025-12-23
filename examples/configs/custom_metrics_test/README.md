# Custom Metrics Test Example

This example demonstrates how to use custom batch metrics with multiple key-value pairs in the experiment manager framework.

## Overview

The custom metrics test shows how to:

- Use the new list format for custom metrics
- Track multiple metrics per batch
- Work with all available trackers (LogTracker, DBTracker, MLflowTracker, TensorBoardTracker, PerformanceTracker)

## Configuration Files

- `env.yaml`: Environment configuration with all trackers enabled
- `experiment.yaml`: Experiment metadata
- `base.yaml`: Base configuration for the pipeline and model
- `trials.yaml`: Trial variations for testing different configurations

## Custom Metrics Tracked

The pipeline tracks the following custom metrics per batch:

- `gradient_norm_mean`: Mean gradient norm across all parameters
- `gradient_norm_std`: Standard deviation of gradient norms
- `gradient_mean_mean`: Mean of gradient means
- `gradient_std_mean`: Mean of gradient standard deviations
- `gradient_min_mean`: Mean of gradient minimums
- `gradient_max_mean`: Mean of gradient maximums
- `batch_accuracy`: Accuracy for this batch
- `output_std`: Standard deviation of model outputs
- `output_mean`: Mean of model outputs
- `loss_std`: Loss value for this batch
- `epoch_batch`: Identifier for epoch and batch
- `learning_rate`: Current learning rate

## Running the Example

```bash
# From the project root
python examples/run_custom_metrics_example.py
```

## Expected Results

After running, you should see:

- Custom metrics in the database (`artifacts/experiment.db`)
- MLflow experiment with custom metrics logged
- TensorBoard logs with custom metrics
- Performance tracker output
- Log files with custom metrics

## Key Features Demonstrated

1. **List Format for Custom Metrics**: Shows how to use the new `Metric.CUSTOM` list format
2. **Multiple Trackers**: Demonstrates that all trackers can handle the new format
3. **Rich Metrics**: Tracks comprehensive gradient and output statistics
4. **Proper Pipeline Structure**: Uses all the correct decorators and patterns

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are installed
2. Check that the configuration files are in the correct location
3. Verify that the workspace directory is writable
4. Check the logs for any error messages

## Related Documentation

- [Pipeline Development Guide](../../../docs/pipeline_development_guide.md) - Complete guide on pipeline implementation
- [Custom Factory Guide](../../../docs/custom_factory_guide.md) - How to create custom factories

---

*Last Updated: December 2024*