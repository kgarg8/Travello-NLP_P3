from utils import consolidateStuff, getImg
from address import getAddress
from models import getModel
from title import getTitle
import time

# these parameter dictionaries show the parameters according to which we train the classifier

params = {'GRAD_CLIP': 100,
          'NAME': 'RNN',
          'SEQ_LENGTH': 1,
          'NUM_EPOCHS': 20,
          'LEARNING_RATE': 0.01,
          'N_HIDDEN': 512,
          'NUM_FEATURES': 9,
          'BATCH_SIZE': 512,
          'NUM_CLUST': 3}

paramsold = {'BATCH_SIZE': 512,
             'GRAD_CLIP': 100,
             'LEARNING_RATE': 0.01,
             'NAME': 'RNN',
             'NUM_CLUST': 3,
             'NUM_EPOCHS': 20,
             'NUM_FEATURES': 8,
             'N_HIDDEN': 512,
             'SEQ_LENGTH': 1,
             'TYPE': '1st classifier'}

paramslstm = {'BATCH_SIZE': 512,
              'GRAD_CLIP': 100,
              'LEARNING_RATE': 0.01,
              'NAME': 'LSTM',
              'NUM_CLUST': 3,
              'NUM_EPOCHS': 10,
              'NUM_FEATURES': 8,
              'N_HIDDEN': 64,
              'SEQ_LENGTH': 4,
              'TYPE': '2nd classifier'}

# here in this example we are fetching a pre-trained model. Note
# that all the saved models are in the ./models directory
try:
    # print paramsold
    rnnModelold = getModel(paramsold, "rnnmodel-old")
except:
    print("couldn't create the model... please correct the error")

try:
    # print paramsold
    rnnModel = getModel(params, "newest")
except:
    print("couldn't create the model... enter a valid filename")

try:
    # print paramslstm
    lstmmodel = getModel(paramslstm, "lstmodel-old")
except:
    print("couldn't create the model... please correct the error")


if __name__ == '__main__':
    # app.run()

    # uncomment these lines to debug your error
    url = "http://www.ladyironchef.com/2017/10/birdfolks-singapore/"
    addresses = getAddress(url, [(paramsold, rnnModelold), (params, rnnModel), (paramslstm, lstmmodel)])
    titles = getTitle(url, addresses)
    images = getImg(url)
    str_to_return = consolidateStuff(url, titles, addresses, images)
