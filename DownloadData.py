# https://github.com/nicolecnolan/MicroBayesAge/releases/tag/data_files

# https://github.com/nicolecnolan/MicroBayesAge/releases/download/data_files/data0.pickle

import urllib.request

def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    print("Downloaded {}%".format(downloaded * 100 // total_size), end="\r", flush=True)

for i in range(114):
    sFilename = "data{}.pickle".format(i)
    print("Downloading", sFilename)
    urllib.request.urlretrieve("https://github.com/nicolecnolan/MicroBayesAge/releases/download/data_files/" + sFilename, "data_files/" + sFilename, show_progress)
    print(flush=True)
