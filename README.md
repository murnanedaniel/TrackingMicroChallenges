# Particle Physics Track Reconstruction Micro-Challenges

Welcome to our open set of "micro-challenges" in particle physics track reconstruction. This project is designed to provide a platform for testing and improving models that attempt to solve various challenges in the field of particle physics track reconstruction.

### MicroChallenges

Each MicroChallenge is a PyTorch Lightning object that contains all the information needed to score its metrics and construct data. These challenges are designed to be self-contained, allowing you to focus on improving your model's performance.

### Models and Configurations

Models that attempt to solve these challenges can be easily integrated into each micro-challenge. Configurations for these models are defined in YAML files, providing a flexible and easy-to-use system for defining and adjusting your models.

### Scoreboard

We have an online scoreboard where you can submit your micro-challenge scores. This provides a competitive aspect to the challenges and allows you to compare your models' performance with others.

### Documentation

We are in the process of writing comprehensive documentation for each challenge, as well as the available models and configurations. This will provide a valuable resource for understanding the challenges and how to best approach them.

### Command Line Interface

Micro-challenges can be run from the command line with the following commands:

- `micro train`: Train your model on a micro-challenge.
- `micro eval`: Evaluate your model's performance on a micro-challenge.
- `micro submit`: Submit your model's score to the online scoreboard.

We hope you find these micro-challenges engaging and useful in improving your models for particle physics track reconstruction. Happy coding!

-------------

## List of MicroChallenges

### Graph Construction

#### 1. Basic Metric Learning

#### 2. Skipped-layer Metric Learning

#### 3. Noisy Metric Learning

#### 4. Shared Hit Metric Learning

#### 5. Fixed Radius Throughput Optimization

#### 6. Module Map Throughput Optimization

------------------

### Edge Classification

#### 1. Basic Edge Classification

#### 2. Shared Hit Classification

#### 3. Generalization Across Curvature

#### 4. Generalization Across Luminosity

#### 5. GNN Throughput Optimization

#### 6. GNN Training Memory Optimization

#### 7. Whole-event Correlation

------------------

### Track Fitting

#### 1. Pointwise Circle Fitting

#### 2. Trackwise Circle Fitting

#### 3. Pointwise Helix Fitting

#### 4. Trackwise Helix Fitting

------------------------

### Track Building

#### 1. Mending Broken Tracks

#### 2. Splitting Merged Tracks

------------------------

### Multi-Task Learning

#### 1. Finding and Fitting Tracks

-------------------------

### From Simulation to Real Data

#### 1. Robustness to Systematic Misalignment

#### 2. Robustness to Random Inefficiency

#### 3. Uncertainty Quantification

--------------------

### Generalist Models

#### 1. Foundation Training with Downstream Fine-Tuning

----------------------


## List of Metrics

### Metric Learning & Track Building: V-Score

### Edge Classification: AUC