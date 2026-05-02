
CUDA_NAME = "cuda:0"
SEEDS = [100, 1000, 2000, 3000, 4000]


REGRESSION_TRAINING = {
    'train_batch_size': 512,
    'test_batch_size': 512,
    'lr': 0.0005,
    'log_interval': 10,
    'num_epochs': 1000,
    'val_interval': 1,
    'early_stop_patience': 50,
}


CLASSIFICATION_TRAINING = {
    'train_batch_size': 256,
    'test_batch_size': 2048,
    'lr': 0.0005,
    'log_interval': 10,
    'num_epochs': 1000,
    'val_interval': 1,
    'early_stop_patience': 50,
}
