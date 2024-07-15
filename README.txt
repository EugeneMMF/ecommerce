This repo contains an E-commerce prediction system.
It utilizes a sentence embedding model from huggingface.

Prerequisites:
--------------

1. sentence_transformers -> pip install sentence_transformers
2. torch -> pip3 install torch
3. tensorflow -> pip3 install tensorflow
4. tf-keras -> pip install tf-keras

Files:
------
customerData.csv is a csv file containing customer data for test purposes
data.csv is a csv file used to generate the embeddings that are later used to get the product recommendations for a customer
embeddings.txt is a text file of the embeddings generated from the data.csv file
main.py is the main script

Operation:
----------
The approach taken was to get the embeddings of the product descriptions then store them for future use.
When getting the recommendations for a customer, we first generate the embeddings of their products and look.
Next we get cosine similarity scores between their embeddings and the stored embeddings.
We then remove any product that they have already purchased.
Next we pick the top <n> products with the highest cosine similarity scores.(<n> was chosen as 10)
These descriptions are what we return

Assumptions:
------------
The product descriptions are unique ie, the product description is the name of the product thus can be substituted for the productID.


How to Use:
-----------
In order to run the file you must first install dependencies.
Then create a csv file containing the customer history of purchases, the columns must have the following minimum columns:
    StockCode,Description,CustomerID
If we are using the precreated embeddings in embeddings.txt and the customer data file customerData.csv and we want the top 5 recommendations you shall then run:
    python3 main.py --customerData "customerData.csv" --embeddings "embeddings.txt" --number 5
If we are using different stock data, then the stock file must have the minimum columns StockCode and Description. Let us call this file data.csv. You shall then run:
    python3 main.py --customerData "customerData.csv" --dataset "data.csv" --number 5
If we want to generate new embeddings and save them we run command:
    python3 generator.py --dataset "data.csv" --embeddings "embeddings.txt"
This will save the embeddings in a file "embeddings.txt"
It is advised that you first generate the embeddings file then run the main script using the embeddings file.
This increases speed. Speed may also be increased using a GPU.