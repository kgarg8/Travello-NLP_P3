from urllib.parse import urlsplit
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import multiprocessing
import multiprocessing
import urllib.request
import urllib.error
import urllib.parse
import json
import os
import re


def process_url(raw_url):
    if ' ' not in raw_url[-1]:
        raw_url = raw_url.replace(' ', '%20')
        return raw_url
    elif ' ' in raw_url[-1]:
        raw_url = raw_url[:-1]
        raw_url = raw_url.replace(' ', '%20')
        return raw_url

# get the images first and then join them later... required if parallelized later


def getImg(url):
    parse_object = urlparse(url)

    opener = urllib.request.build_opener()
    opener.addheaders = [
        ('User-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/50.0.2661.102 Chrome/50.0.2661.102 Safari/537.36')]

    urlcontent = opener.open(url).read()
    soup = BeautifulSoup(urlcontent, "lxml")
    images = soup.findAll("img")
    imgurls = re.findall('img .*src="(.*?)"', urlcontent)

    collected_images = []

    for image in images:
        try:
            imgurl = re.findall('img .*src="(.*?)"', str(image))[0]
            if imgurl[-3:] != "svg":
                imgurl = process_url(imgurl)

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


if __name__ == '__main__':
    url = input("enter website to get images from\n")
    images = getImg(url)
    print(images)
