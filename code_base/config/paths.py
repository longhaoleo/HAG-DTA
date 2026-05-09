import os


DATA_ROOT = os.environ.get('HAG_DTA_DATA_ROOT', '../data')
CACHE_ROOT = os.environ.get('HAG_DTA_CACHE_ROOT', '/root/autodl-tmp/HAG-DTA-cache')
PROCESSED_DIR = os.environ.get('HAG_DTA_PROCESSED_DIR', os.path.join(CACHE_ROOT, 'processed'))
FOLD_DIR = os.environ.get('HAG_DTA_FOLD_DIR', os.path.join(CACHE_ROOT, 'fold_indices'))

OUTPUT_ROOT = os.environ.get('HAG_DTA_OUTPUT_ROOT', '/root/autodl-tmp/HAG-DTA-runs')
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, 'checkpoints')


def raw_data_dir(dataset):
    return os.path.join(DATA_ROOT, dataset)


def processed_file(dataset_name):
    return os.path.join(PROCESSED_DIR, dataset_name + '.pt')


def fold_file(dataset, fold_id):
    return os.path.join(FOLD_DIR, f'{dataset}_fold{fold_id}.json')


def output_file(filename):
    return os.path.join(OUTPUT_ROOT, filename)


def ensure_runtime_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FOLD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
