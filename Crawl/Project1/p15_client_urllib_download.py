import urllib.parse
import urllib.request


url="http://127.0.0.1:5000"
try:
    html=urllib.request.urlopen(url)
    html = html.read()
    fileName=html.decode()
    print("准备下载:"+fileName)
    urllib.request.urlretrieve(url + "?fileName=" + urllib.parse.quote(fileName), "download"+fileName)
    # 上一句urllib.request中专门有一个urlretrieve函数是从服务器获取文件保存到本地，可代替下方代码
    # data = urllib.request.urlopen(url + "?fileName=" + urllib.parse.quote(fileName))
    # data = data.read()
    # fobj=open("download "+fileName,"wb")
    # fobj.write(data)
    # fobj.close()
    print("下载完毕")

except Exception as err:
    print(err)
