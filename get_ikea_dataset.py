import re

import bs4
import requests


def get_allcaps_words(s):
    for w in re.findall("\w{2,}", s, flags=re.UNICODE):
        w = w.strip("- ")
        if w == w.upper():
            yield w


sess = requests.session()
for letter in range(27):
    url = f"https://www.ikea.com/fi/fi/catalog/productsaz/{letter}/"
    data = requests.get(url).text
    soup = bs4.BeautifulSoup(data, features="html.parser")

    for node in soup.select("li.productsAzLink a"):
        for word in get_allcaps_words(node.text):
            print(word)
