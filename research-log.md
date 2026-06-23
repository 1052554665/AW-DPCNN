To prevent the higher accuracy in confusion matrix, the dataset was reconstructed as follows.

- the windows length was set to 4096, and the hop length was set to 2048.


python scripts/build_cwru_de.py --output-dir ./datasets/cwru_de --file-split 60,20,20 --split-seed 42 --metadata --verify --workers 16 --win-len 4096 --hop-len 2048
