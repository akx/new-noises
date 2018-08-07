import requests
import bs4

data = requests.get("http://everynoise.com/everynoise1d.cgi?scope=all").text
soup = bs4.BeautifulSoup(data, features="html.parser")

for node in soup.select("td.note a"):
    print(node.text)
