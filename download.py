import gdown

url = "https://drive.usercontent.google.com/download?id=1d0SfbGkhNs7Xt6qIuwmX3NlE_zW8xp2K&export=download&authuser=2&confirm=t&uuid=b192e63b-7b48-4db4-a39d-bf1f8a2693a3"
output = "model_best.pth"
gdown.download(url, output, quiet=False)
