import gdown

url = "https://drive.usercontent.google.com/download?id=1RPYTREPZ_PVzOHnESgTdg8x7MjVMl_4c&export=download&authuser=0&confirm=t&uuid=e6148024-cd81-4ddf-a2bc-a7c2da29a55a"
output = "3-gram.arpa"
gdown.download(url, output, quiet=False)

url = "https://drive.usercontent.google.com/download?id=1NF7tKEBjIsLYqd2TtMSx6R9hr-mMTYGK&export=download&authuser=0&confirm=t&uuid=b4fb86ad-7118-4a92-b51c-42b86c178b1e"
output = "librispeech-vocab.txt"
gdown.download(url, output, quiet=False)

url = "https://drive.usercontent.google.com/download?id=1SXOMnmVDEf9lua6qY8J8yW1-xbTDOND6&export=download&authuser=0&confirm=t&uuid=f073631e-f0d7-438e-ad87-44ec64585c45"
output = "model_best.pth"
gdown.download(url, output, quiet=False)

url = "https://drive.usercontent.google.com/download?id=1MdWJSj80o40E9sU4wDw1pk7dFWjXnT1v&export=download&authuser=0&confirm=t&uuid=61f7318c-c95c-440c-8562-df9347dfc540"
output = "config.json"
gdown.download(url, output, quiet=False)
