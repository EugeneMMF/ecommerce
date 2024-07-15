import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import json
from argparse import ArgumentParser
import torch

def load_dataset(csv_file:str):
    try:
        return pd.read_csv(csv_file, encoding="unicode_escape")
    except FileNotFoundError as e:
        print(repr(e), csv_file)

def verify_keys(df:pd.DataFrame, keys:list[str]):
    df_keys = df.keys()
    for key in keys:
        if key not in df_keys:
            return False
    return True

def groupByCustomerID(data:pd.DataFrame):
    groupings = {}
    index = {val:i for i,val in enumerate(data.keys())}
    for d in data.values:
        customerID = d[index['CustomerID']]
        description = d[index['Description']]
        product = d[index['ProductID']]
        if groupings.get(customerID):
            groupings[customerID].append({"Description": description, "Product": product})
        else:
            groupings[customerID] = [{"Description": description, "Product": product},]
    return {id:pd.DataFrame(dat) for id,dat in groupings.items()}

def evaluateTopPicks(model, customerData:pd.DataFrame, embeddings:dict[str,np.ndarray], number:int=5):
    descriptions = list(customerData['Description'].values)
    to_pop = []
    for i, desc in enumerate(descriptions):
        if not isinstance(desc, str):
            to_pop.append(i)
    for i in reversed(to_pop):
        descriptions.pop(i)
    averageEmbedding = model.encode(descriptions)
    # averageEmbedding = np.average(averageEmbedding, axis=0)
    results = {}
    all_results = {}
    embeddingValues = list(embeddings.values())
    for avgEmbed in averageEmbedding:
        avgEmbed = np.tile(avgEmbed, (len(embeddingValues),1))
        results = util.cos_sim(torch.Tensor(embeddingValues), torch.Tensor(avgEmbed))
        results = {desc:val[0] for desc,val in zip(embeddings.keys(), results) if val[0] != 1}
        results = sorted(results.items(), key=lambda x:x[1], reverse=True)
        if all_results == {}:
            all_results = dict({i[0]:i[1] for i in results})
        else:
            i = 0
            j = 0
            while i < number:
                if j >= len(results):
                    break
                if results[j][0] in all_results.keys():
                    all_results[results[j][0]] = max(results[j][1], all_results[results[j][0]])
                    j+=1
                    continue
                else:
                    all_results[results[j][0]] = results[j][1]
                    i += 1
                    j += 1
    keys = list(all_results.keys())
    for i in reversed(range(len(keys))):
        key = keys[i]
        if key in descriptions:
            all_results.pop(key)
    all_results = sorted(all_results.items(), key=lambda x:x[1], reverse=True)
    return dict(all_results[:number])

def generateEmbeddings(model, dataset:pd.DataFrame, filename:str|None=None):
    descriptions = list(dataset['Description'].values)
    productIDs = list(dataset['ProductID'].values)
    customerIDs = list(dataset['CustomerID'].values)
    to_pop = []
    for i, desc in enumerate(descriptions):
        if not isinstance(desc, str):
            to_pop.append(i)
    for i in reversed(to_pop):
        descriptions.pop(i)
        productIDs.pop(i)
        customerIDs.pop(i)
    embeddings = model.encode(descriptions)
    results = []
    for productID,description,embedding in zip(productIDs,descriptions,embeddings):
        results.append({"ProductID": productID, "Description": description, "Embedding": embedding.tolist()})
    if filename:
        with open(filename, 'w') as f:
            for data in results:
                json.dump(data, f)
                f.write("\n")
    return results

def getEmbeddings(filename:str):
    results = []
    with open(filename, 'r') as f:
        for line in f:
            results.append(json.loads(line))
            results[-1]['Embedding'] = np.array(results[-1]['Embedding'])
    return results

def main(model, customerData:str, *, dataset:str|None=None, embeddingsFile:str|None=None, number:int=10):
    keys = ["Description", "ProductID"]
    if dataset:
        data = load_dataset(dataset)
        if not verify_keys(data, keys):
            raise Exception("Dataset does not contain the correct columns.")
        fullEmbeddings = generateEmbeddings(model, data)
    elif embeddingsFile:
        fullEmbeddings = getEmbeddings(embeddingsFile)
    else:
        raise Exception("Either provide data or use a presaved embeddings file.")
    embeddings = {val['Description']:val['Embedding'] for val in fullEmbeddings}
    customer_data = load_dataset(customerData)
    keys = ['Description', 'ProductID', 'CustomerID']
    if not verify_keys(customer_data, keys):
        raise Exception("Customer data does not contain the correct columns.")
    return evaluateTopPicks(model, customer_data, embeddings, number)

if __name__ == "__main__":
    argparser = ArgumentParser()
    model = SentenceTransformer("all-MiniLM-L6-v2").to('cuda' if torch.cuda.is_available() else 'cpu')
    argparser.add_argument("--customerData", type=str, default=None, help="The .csv file name that contains the customer data.\n Must be formatted with the columns \"StoreCode\", \"Description\" and \"CustomerID\"")
    argparser.add_argument("--dataset", type=str, default=None, help="The .csv file name that contains the stock data.\n Must be formatted with the columns \"StoreCode\", \"Description\"")
    argparser.add_argument("--embeddings", type=str, default=None, help="The file containing the embeddings this is generated by the function generateEmbeddings")
    argparser.add_argument("--number", type=int, default=10, help="The number of recommendations to give.")
    args = argparser.parse_args()
    top_picks = main(model, customerData=args.customerData, dataset=args.dataset, embeddingsFile=args.embeddings, number=args.number)
    print(f"The top {args.number} picks were: ")
    print(top_picks.keys())