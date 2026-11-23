import urllib.request

IMAGENET_LABELS_FILE = "imagenet_classes.txt"

def load_imagenet_labels():
    if not os.path.exists(IMAGENET_LABELS_FILE):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        urllib.request.urlretrieve(url, IMAGENET_LABELS_FILE)
    with open(IMAGENET_LABELS_FILE, "r") as f:
        return [line.strip() for line in f.readlines()]
