import opendatasets as od

# This will force the terminal to ask you for the ID and Key
dataset_url = 'https://www.kaggle.com/datasets/jessicali9530/celeba-dataset'

print("Look at the terminal below! It will ask for your Username and Key.")
od.download(dataset_url)