import urllib.request
import urllib.parse

url = "http://127.0.0.1:5000"

try:
    province = urllib.parse.quote("广东")
    city = urllib.parse.quote("深圳")
    data = "province=" + province + "&city=" + city
    html = urllib.request.urlopen(url + "?" + data)
    html = html.read()
    html = html.decode()
    print(html)

except Exception as err:
    print(err)
