# IVU course project


## Optimizer Benchmark on example of Rice grain image classification task
This project benchmarks various optimization algorithms on a rice grain image classification task using a simple small CNN.

### Objective
Evaluate and compare the performance of multiple optimizers in terms of:
- Validation accuracy
- Validation loss
- Gradient and weight norm tracking
- GPU memory usage

### Tested Optimizers
- SGD (no momentum)
- SGD with momentum
- Adam
- AdamW
- Adagrad

### Project Structure
Each optimizer is evaluated using the same model [SimpleCNN](./model.py) and data pipeline. The results **(accuracy, f1, precision, recall, loss, and weight norm)** are logged using TensorBoard.

### Usage
Create environment:
```bash
conda env create -f environment.yml
```

Run training in the [notebook](TrainingPipeline.ipynb) or:
```bash
python run_experiments.py \
  --data_dir ./rice_images \
  --results_dir ./results \
  --img_size 64 \
  --batch_size 64 \
  --epochs 7 \
  --seed 52
```
Metrics will be printed to the console


Run **tensorboard** in separate console to see logs:
```bash
tensorboard --logdir=./results --host localhost --port 8123
```



