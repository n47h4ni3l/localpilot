#!/usr/bin/env python
import yaml
from localpilot.model import Model
from localpilot.dataset import Dataset

# Load configuration
with open('evaluation/evaluation_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize model and dataset
model = Model()
dataset = Dataset(config['dataset'])

# Benchmark the model
results = model.benchmark(dataset, config['parameters']['tasks'])

# Save benchmark results
with open('evaluation/benchmark_results.json', 'w') as file:
    json.dump(results, file)
