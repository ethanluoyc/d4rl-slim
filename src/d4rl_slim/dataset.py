# ruff: noqa: E501
import os
import urllib.request
import uuid

try:
    import h5py
    from tqdm import tqdm
except ImportError:
    h5py = None

    def tqdm(x, desc):
        return x


from d4rl_slim import infos

DATASET_PATH = None


def set_dataset_path(path):
    global DATASET_PATH
    DATASET_PATH = path
    os.makedirs(path, exist_ok=True)


set_dataset_path(
    os.environ.get("D4RL_DATASET_DIR", os.path.expanduser("~/.d4rl/datasets"))
)


def _get_keys(h5file):
    if h5py is None:
        raise ImportError("Please install `h5py` to use the dataset features")
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def _filepath_from_url(dataset_url):
    _, dataset_name = os.path.split(dataset_url)
    dataset_filepath = os.path.join(DATASET_PATH, dataset_name)
    return dataset_filepath


def download_dataset_from_url(dataset_url):
    dataset_filepath = _filepath_from_url(dataset_url)
    if not os.path.exists(dataset_filepath):
        print("Downloading dataset:", dataset_url, "to", dataset_filepath)
        dataset_tmppath = dataset_filepath + ".tmp.{}".format(uuid.uuid4())
        urllib.request.urlretrieve(dataset_url, dataset_tmppath)
        os.rename(dataset_tmppath, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath


def get_dataset(dataset_name: str):
    if h5py is None:
        raise ImportError("Please install `h5py` to use the dataset features")
    dataset_url = infos.get_dataset_url(dataset_name)
    h5path = download_dataset_from_url(dataset_url)

    data_dict = {}
    with h5py.File(h5path, "r") as dataset_file:
        for k in tqdm(_get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    # Run a few quick sanity checks
    for key in ["observations", "actions", "rewards", "terminals"]:
        assert key in data_dict, "Dataset is missing key %s" % key

    return data_dict