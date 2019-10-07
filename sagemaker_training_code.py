import sagemaker
from sagemaker.pytorch import PyTorch

role = sagemaker.get_execution_role()

metrics = [
    {'Name': 'train:loss', 'Regex': 'train_loss=(.*?);'},
    {'Name': 'train:accu', 'Regex': 'train_accu=(.*?);'},
    {'Name': 'val:loss', 'Regex': 'val_loss=(.*?);'},
    {'Name': 'val:accu', 'Regex': 'val_accu=(.*?);'},
]

hyperparameters = {
    "epochs": 3,
    "lr": 0.001,
    "dropout": 0.1,
    "batch-size": 128,
    "num-workers": 2,
    "train-data-file": "mnist_train.csv",
    "valid-data-file": "mnist_test.csv",
    "no-cuda": True,
    "sagemaker-logging": True,
    "verbose": True
}

train_bucket = "s3://sagemaker-corvin-mnist/data"
store_location = "s3://sagemaker-corvin-mnist/models"

pt_estimator = PyTorch(entry_point="train.py",
                       source_dir='mnist-pytorch-sagemaker',
                       framework_version="1.0.0",
                       train_instance_type='ml.m5.large',
                       role=role,
                       train_instance_count=1,
                       metric_definitions=metrics,
                       output_path=store_location,
                       hyperparameters=hyperparameters)

pt_estimator.fit(inputs={"training": train_bucket})
