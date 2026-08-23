#!/usr/bin/env python
import yaml
from localpilot.model import Model
from localpilot.dataset import Dataset

# Load configuration
with open('training/training_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize model and dataset
model = Model(config['parameters']['adapter_type'])
dataset = Dataset(config['dataset'])

# Train the model
model.train(dataset, config['parameters'])

# Save the trained model
model.save('models/adapter_weights')
