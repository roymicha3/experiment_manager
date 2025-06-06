# Matplotlib Tick Issue Fix for Performance Tracker

## Problem

When running `python examples/performance_tracker_example.py`, users encountered the following error:

```
Locator attempting to generate 2312640 ticks ([19441.925, ..., 21047.924305555556]), 
which exceeds Locator.MAXTICKS (1000).
```

## Root Cause

The error occurred in the `generate_performance_plot()` method of the `PerformanceTracker` class when:

1. **High-frequency data collection**: The tracker was sampling every 0.5 seconds
2. **Large history buffer**: Keeping 2000 snapshots in memory  
3. **Fixed tick locator**: Using `mdates.MinuteLocator(interval=1)` on dense time-series data
4. **No data decimation**: Plotting all raw data points without reducing density

When plotting thousands of data points with fine-grained timestamps, matplotlib attempted to generate millions of ticks, exceeding the default `MAXTICKS` limit of 1000.

## Solution

### 1. **Intelligent Data Decimation**
```python
# Limit to 1000 data points for plotting
max_points = 1000
if len(snapshots) > max_points:
    step = len(snapshots) // max_points
    indices = list(range(0, len(snapshots), step))
    if indices[-1] != len(snapshots) - 1:
        indices.append(len(snapshots) - 1)  # Always include the last point
    snapshots = [snapshots[i] for i in indices]
```

### 2. **Smart Time-Based Tick Locators**
```python
duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else timedelta(0)

if duration.total_seconds() < 3600:  # Less than 1 hour
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, int(duration.total_seconds() / 300))))
elif duration.total_seconds() < 86400:  # Less than 1 day
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    interval = max(1, int(duration.total_seconds() / 3600 / 6))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
else:  # More than 1 day
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

# Safety net: Limit maximum number of ticks
ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
```

### 3. **Optimized Default Configuration**
```python
# In examples/performance_tracker_example.py
tracker = PerformanceTracker(
    monitoring_interval=1.0,        # Increased from 0.5s
    history_size=500,              # Reduced from 2000
    # ... other settings
)
```

### 4. **Reduced Simulation Load**
```python
# Reduced work duration from 800ms to 300ms
while time.time() - start_time < 0.3:
    _ = sum(i * i for i in range(500))  # Reduced computation
```

## Benefits

✅ **Prevents matplotlib overflow**: No more MAXTICKS errors  
✅ **Maintains data integrity**: Key points (start/end) always preserved  
✅ **Adaptive formatting**: Time axis format adjusts to data duration  
✅ **Better performance**: Faster plotting with decimated data  
✅ **Preserved visualization quality**: Still shows meaningful trends  
✅ **Backwards compatible**: No API changes required  

## Testing

After applying the fix:
- ✅ `python examples/performance_tracker_example.py` runs successfully
- ✅ Performance plots are generated without errors
- ✅ Visualization quality is maintained
- ✅ No breaking changes to existing code

## Files Modified

1. `experiment_manager/trackers/plugins/performance_tracker.py` - Fixed plotting method
2. `examples/performance_tracker_example.py` - Optimized configuration

## Prevention

To avoid similar issues in the future:
- Always consider data density when plotting time-series
- Implement decimation for large datasets
- Use adaptive tick locators based on data duration
- Set reasonable limits on in-memory data storage
- Test with various data sizes and time ranges 