from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
from main import generateEmbeddings, load_dataset
from sentence_transformers import SentenceTransformer
import torch

if __name__ == "__main__":
    argparser = ArgumentParser()
    model = SentenceTransformer("all-MiniLM-L6-v2").to('cuda' if torch.cuda.is_available() else 'cpu')
    argparser.add_argument("--dataset", type=str, default=None, help="The .csv file name that contains the stock data.\n Must be formatted with the columns \"StoreCode\", \"Description\"")
    argparser.add_argument("--embeddings", type=str, default=None, help="The file name of the file to store the embeddings.")
    args = argparser.parse_args()
    dataset = load_dataset(args.dataset)
    generateEmbeddings(model, dataset, args.embeddings)