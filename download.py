import gdown

url = "https://drive.usercontent.google.com/download?id=1d0SfbGkhNs7Xt6qIuwmX3NlE_zW8xp2K&export=download&authuser=2&confirm=t&uuid=5dce46a0-c18d-4fc5-8f1d-0511fda9802d"
output = "model_best.pth"
gdown.download(url, output, quiet=False)
