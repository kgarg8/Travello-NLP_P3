from sklearn import preprocessing
from fuzzywuzzy import process
from sklearn.cluster import KMeans
from bs4 import BeautifulSoup
import numpy as np
import urllib.request
import urllib.error
import urllib.parse
import json
import re
from create_training import getvec
import logging
import shutil
import os
import torch
import pdb


eps = 0


def parsePage(url):
    '''
    This method is very important and it used at many places in the package. It is used to
    parse a page and extract relevant information from the page ie. the text and it creates
    two other data structures which are helpful for other computations all throughout

    Parameters
    ----------
    url : The url of the page

    Returns
    -------
    soup : It is a BeautifulSoup object which has parsed the webpage are separated it based
        on the html tags. For more information please refer to the BeautifulSoup library

    paragraphs : It is a list of all the paragraphs on the webpage. Paragraphs are pieces
        of text which are separated by '\n'. Throughout the documentation we have used the
        term index of a paragraph which is nothing but it's index in this list

    paradict : It is a python dictionary which stores the reverse of paragraphs ie. the
        indices are referred to by the paragraphs which they index to
    '''

    opener = urllib.request.build_opener()
    # this header is important as many websites detect that the request is coming from
    # a python bot and they reject the request. This header is to make the request look
    # as if it's coming from an authentic browser
    opener.addheaders = [
        ('User-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/50.0.2661.102 Chrome/50.0.2661.102 Safari/537.36')]
    response = opener.open(url)
    page = response.read()
    soup = BeautifulSoup(page, 'lxml')

    # remove the styling and scripting tags
    for elem in soup.findAll(['script', 'style']):
        elem.extract()

    raw = soup.get_text().encode('ascii', 'ignore')

    # page_title is the title of the page which is not present in raw
    page_title = soup.select("title")[0].get_text().encode(
        'ascii', 'ignore').strip().encode('ascii')

    paragraphs = []

    for p in raw.split('\n'):
        p = p.strip()
        if len(p) > 2:
            p = p.replace('\t', '')
            p = p.replace('\n', '')
            paragraphs.append(p)

    paragraphs.append(page_title)
    paradict = {}
    for i in range(len(paragraphs)):
        if paragraphs[i] not in paradict:
            paradict[paragraphs[i]] = i

    # special entry for the page_title at the end of the list
    paradict[page_title] = -1

    return soup, paragraphs, paradict


def consolidateStuff(url, titles, addresses, images):
    '''
    This method is used to 'glue' the correct titles, addresses, write-ups
    and the images to form a blob of complete information about a particular point
    of interest in the webpage

    Parameters
    ----------
    url : The url of the page

    titles : The data structure as returned by the getTitles method

    addresses : The addresses as returned by the getAddress function. Note that
        these addresses are already consolidated amongst themselves

    images : The url of images as returned by the getImage method.

    Returns
    -------
    jsonoutput : A stringified dictionary where indices(starting from 0) repsesent another
        dictionary indexed by 'Place Name', 'Write-up', 'Address' and 'Image URL'

    '''
    soup, paragraphs, paradict = parsePage(url)
    lens = [len(p) for p in paragraphs]
    jsonoutput = {}
    posspara = LongParas(lens)
    # print type(posspara)
    titles = [paradict[t] for t in titles]

    # special consilidation for TripAdvisor
    if 'tripadvisor' in url:
        jsonoutput[0] = {'Place Name': paragraphs[-1],
                         'Write-up': paragraphs[posspara[0]],
                         'Address': addresses
                         }

    else:
        addrs = []

        # the head of addresses as explained in the getTitles method
        for address in addresses:
            addrs.append(paradict[address[0]])

        addrs = np.array(addrs)

        # to glue the titles, addresses and the write-ups to form a correct
        # and complete blob of information
        fullThing = getFull(titles, addrs, posspara)

        for i in range(len(fullThing)):
            onething = fullThing[i]
            jsonoutput[i] = {'Place Name': paragraphs[onething[0]],
                             'Write-up': paragraphs[posspara[onething[1]]],
                             'Address': addresses[onething[2]]
                             }

    # choices is the array of stringified image tags which have everything from
    # image-src to image-alt. Hope is that it will give a hint as to which place that
    # image belongs to and using string similarity algorithms we will assign
    # that placename to this image

    choices = [str(image).decode('utf-8') for image in images]

    if len(choices) == 0:
        jsonoutput[i]['Image URL'] = "http://herbookthoughts.reads-it.com/wp-content/uploads/2014/6/6a1143f571184db25f94613edd43b40af6d3a629221aba00d9efdcfef5efd84.jpg"

    else:
        for i in range(len(titles)):
            # extractOne returns a tuple (imageName, similarity_score)
            rightImage = process.extractOne(jsonoutput[i]['Place Name'],
                                            choices)[0]

            # we want the src attribute only
            imgurls = re.findall('img .*src="(.*?)"', rightImage)

            jsonoutput[i]['Image URL'] = imgurls[0]

    # print jsonoutput
    return json.dumps(jsonoutput, indent=4)


def LongParas(lens):
    '''
    This method returns the long write-ups in the webpage by clustering them into 2
    clusters based on their lengths. Idea is that kmeans will separate the longer
    paragraphs

    Parameters
    ----------
    lens : A list where each element represents the length of that corresponding
        paragraphs at that index

    Returns
    -------
    posspara : A list of indices of the long paragraphs
    '''

    res = np.array(lens)
    # reshapre the res array to form a matrix of size (1, len(res))
    res = res.reshape(-1, 1)
    est = KMeans(n_clusters=2)
    est.fit(res)
    labels = est.labels_
    # since the cluster numbers are randomly assigned, we need to find the cluster number
    # the long paragraphs by finding out in which cluster the longest
    # paragraph lies
    bestpara = np.argmax(res)
    reqlabel = labels[bestpara]

    # these are the possible paragraphs about the places of interest
    posspara = np.where(labels == reqlabel)[0]

    return posspara


def findmin(arr):
    '''
    This is a simple method to find the least non-negative number in an array. Could'nt
    find an easier way to do this task and therefore had to write it in a new method

    Parameters
    ----------
    arr : A numpy array

    Returns
    -------
    The smallest non-negative number in arr
    '''

    maxx = np.max(arr)
    for i in range(len(arr)):
        if arr[i] < 0:
            arr[i] = maxx + 1
    return np.argmin(arr)


def getFull(headers, addresses, possparas):
    '''
    A very small but important method which for every header(take note) and not the
    address finds the appropriate address object and the write-up by using loaclity
    arguments described elsewhere. Now we can appreciate the passing of header and address
    information by their indices rather than the text itself

    Parameters
    ----------
    headers : A list of indices of all the header paragraphs

    addresses : A list of indices of all the first line of addresses... only first
        line is needed as we will make locality arguments

    possparas : A list of indices of all the write-ups

    Returns
    -------
    blob : A list of shape=(number_of_pts_of_interests, 3) (Hopefully!)
        Each element of blog is itself a list which has the indices of the header,
        it's corresponding write-up and address
    '''
    out = []
    for header in headers:
        parapos = findmin(possparas - header)
        addrpos = findmin(addresses - header)
        out.append([header, parapos, addrpos])
    blob = np.array(out)
    return blob


def process_url(raw_url):
    '''
    Method to replace the ' ' with '%20' as is the norm in urls

    Parameters
    ----------
    raw_url : raw URL to be standardized

    Returns
    -------
    raw_url : The modified URL
    '''

    if ' ' not in raw_url[-1]:
        raw_url = raw_url.replace(' ', '%20')

    elif ' ' in raw_url[-1]:
        raw_url = raw_url[:-1]
        raw_url = raw_url.replace(' ', '%20')

    return raw_url


def getImg(url):
    '''
    This function retrieves all the images on the webpage which are larger than 80x80
    ie. icon size because larger images tend to be associated with a place-name.
    TODO - Parallelize this method for better performance

    Parameters
    ----------
    url : The url of the page

    Returns
    -------
    images : An array of strings where each element is the <img> tag but stringified
    '''
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ('User-agent',
         'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/50.0.2661.102\ Chrome/50.0.2661.102 Safari/537.36')]

    urlcontent = opener.open(url).read()
    soup = BeautifulSoup(urlcontent, "lxml")

    # use soup to extract all the image tags
    images = soup.findAll("img")

    collected_images = []

    for image in images:
        try:
            imgurl = re.findall('img .*src="(.*?)"', str(image))[0]
            # reject .svg images
            if imgurl[-3:] != "svg":
                imgurl = process_url(imgurl)

                # some pages have img tags where height and width attributes are missing
                # so to choose valid images from that page we have to download all the images
                # and choose those which have a length > 5000(works well in practice)
                # Problem is that it can take time if the images are large in size

                if 'height' in str(image) and 'width' in str(image):
                    if int(image['height']) > 80 and int(image['width']) > 80:
                        collected_images.append(image)
                        # print (imgurl, image["alt"], image['height'], image['width'])

                else:
                    imgdata = urllib.request.urlopen(imgurl).read()
                    if len(imgdata) > 5000:
                        collected_images.append(image)
                        # print (image, len(imgdata))

        except:
            pass
    return collected_images


# Can return the data in required shape
def getData(paragraphs, NUM_FEATURES, BATCH_SIZE, SEQ_LENGTH=None):
    '''
    This method converts the webpage into a feature matrix which is then
    be used by the classifier to find out addresses. It's sole use in the package
    is in the getAddress method.

    Parameters
    ----------
    paragraphs : The list of paragraphs on the webpage

    NUM_FEATURES : Number of features we want to consider for a single paragraph.
        Usually filled in by params['NUM_FEATURES']

    BATCH_SIZE : Most classifiers usually take in input vectors in a batch as it is
        faster. This parameter is fixed by params['BATCH_SIZE']

    SEQ_LENGTH : The Recursive NN and the Long Short Term NN predict the label of a
        data point by looking at some previous and some next data points.
        This parameter decides how for into the past and the future should we look

    Returns
    -------
    data : an array of shape=(batches, SEQ_LENGTH, NUM_FEATURES)
    '''

    len1 = len(paragraphs)

    if SEQ_LENGTH:
        batches = len1 / (BATCH_SIZE * SEQ_LENGTH) + 1

        # this auxillary array is used an intermediate while reshaping the data. Right
        # now the sequences are not separated and in the second pass we will separate it
        # bad design... can be improved later

        data1 = np.zeros((BATCH_SIZE * (batches) * SEQ_LENGTH, NUM_FEATURES))

        # we fill in data1 with feature vectors generated by the getvec() method
        # in create_training file
        for i in range(len1):
            data1[i] = np.array(getvec([paragraphs[i]])[:NUM_FEATURES])

        # bug here... Forgot to pad with 0s at the starting and the end
        # but still it is working fine with SEQ_LENGTH = 1 we
        # need no padding there

        # the real array ie. with proper shape, which is to be returned
        data = np.zeros((BATCH_SIZE * (batches), SEQ_LENGTH, NUM_FEATURES))
        for i in range(len(data1)):
            data[i / SEQ_LENGTH, i % SEQ_LENGTH, :] = data1[i]

        del(data1)

    # if SEQ_LENGTH is not used then it means that we are using a MLP and not RNN or LSTM
    else:
        batches = len1 / BATCH_SIZE + 1
        data = np.zeros((batches * BATCH_SIZE, NUM_FEATURES))
        for i in range(len1):
            data[i / BATCH_SIZE,
                 :] = np.array(getvec([paragraphs[i]])[:NUM_FEATURES])

    return data


def getScores(pred, paragraphs, params):
    '''
    This is an auxillary method used to print the scores assigned by the classifier
    to the paragraphs. Currently it only supports only RNN and LSTM.

    Parameters
    ----------
    pred : The Theano function which can predict the labels of paragraphs

    paragraphs : The list of paragraphs on the webpage

    params : The parameters needed by the model in the form of a dictionary

    Returns
    -------
    out : A list of tuples which has a paragraph and it's score as predicted by the
        classifier
    '''

    X = getData(paragraphs, params['NUM_FEATURES'], params[
                'BATCH_SIZE'], SEQ_LENGTH=params['SEQ_LENGTH'])

    res = pred(X).flatten()
    out = []
    for i in range(len(paragraphs)):
        out.append((paragraphs[i], res[i]))
    return out


def load_dataset(X, y, NUM_FEATURES, wndw=1):
    '''
    This method takes in the data and breaks it into training and validation data.
    Validation data consists of the last 1000 samples of the data
    Then it buffers them with 0 vectors at the front and at the back depending on SEQ_LENGTH

    Parameters
    ----------
    X : A list of shape=(n_samples, n_features)

    y : A list of shape=(n_samples,)
        The target labels for the paragraphs

    NUM_FEATURES : The number of features to be considered starting from feature 0

    SEQ_LENGTH : The window_size for buffering the input with 0 vectors

    Returns
    -------
    X_train : Training data points

    y_train : Labels of training data points

    X_val : Validation data points

    y_val : Labels of validation data points
    '''
    for i in range(len(X)):
        X[i] = np.array(X[i])

    # X = preprocessing.scale(X)
    # poly = PolynomialFeatures(degree = 2)
    # X = poly.fit_transform(X)
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int32')

    # normalize the continuous valued columns except the 5th column
    X = X.T
    X[[0, 1, 2, 3, 5, 6]] = preprocessing.scale(X[[0, 1, 2, 3, 5, 6]])
    X = X[:NUM_FEATURES]
    X = X.T

    val_split = round(len(X) * 0.2)
    X_train = X[:-val_split]
    y_train = y[:-val_split]

    X_val = X[-val_split:]
    y_val = y[-val_split:]
    # the cross-categorical cost function requires input in the range (0,1)
    # X[X < eps] = eps
    # X[X > 1 - eps] = 1 - eps

    if wndw / 2 > 0:
        num_feat = len(X[0])
        # 0 buffers for training and validation data...
        # not needed for labels though as buffering doesn't increase the
        # number of data points... this a bug till now
        Xbuffer = np.ones((wndw // 2, num_feat)) * eps
        X_train = np.vstack([Xbuffer, X_train, Xbuffer])
        X_val = np.vstack([Xbuffer, X_val, Xbuffer])

    return X_train, y_train, X_val, y_val


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file 'log_path' """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s:%(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves checkpoint + 'best.pth.tar'"""

    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print('Checkpoint directory does not exist! Making directory {}'.format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(filepath, model, optimizer=None):
    """Loads model parameters (state_dict) from filepath. If optimizer is provided, loads state_dict of optimizer assuming it is present in filepath."""

    if not os.path.exists(filepath):
        print('File does not exist {}'.format(filepath))
        return

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file"""

    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
