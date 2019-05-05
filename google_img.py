from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time 
import urllib.request
from bs4 import BeautifulSoup as bs
import re
import os

base_url_part1='https://www.google.com/search?q='
base_url_part2='&source=lnms&tbm=isch'

search_query=input("Input what you want to search\n")

class Google_crawler:
    def __init__(self):
       self.url= base_url_part1+search_query+base_url_part2
    
    def start_brower(self):
        #chrome_options=Options()
        #chrome_options.add_argument("--disable-infobars")

        driver=webdriver.Chrome()
        driver.maximize_window()
        driver.get(self.url)
        return driver
    def downloadImg(self,driver):
        t=time.localtime(time.time())
        foldername=str(t.__getattribute__("tm_year"))+ "-" +str(t.__getattribute__("tm_mon"))+ "-" +\
            str(t.__getattribute__("tm_mday"))

        localpath='C:/temp/%s' %(foldername)

        if not os.path.exists(localpath):
            os.makedirs(localpath)

        img_url_down={}
        x=0
        pos=0

        for i in range(1):
            pos+=500
            js="document.documentElement.scrollTop=%d" %pos
            driver.execute_script(js)
            time.sleep(2)
            html_page=driver.page_source

            soup=bs(html_page,"html_parser")
            imglist=soup.find_all('img',{'class':'rg_ic rg_i'})
            for imgurl in imglist:
                try:
                    print(x,end=' ')
                    if imgurl['src'] not in img_url_down:
                        target='{}/{}.jpg'.format(localpath,x)
                        img_url_down[imgurl['src']]=''
                        urllib.request.urlretrieve(imgurl['src'],target)
                        time.sleep(1)
                        x+=1
                except KeyError:
                    print("ERROR!")
                    continue

        
    def run(self):
        print("Start\n")
        driver=self.start_brower()
        self.downloadImg(driver)
        driver.close()
        print("Download has finished.\n")

if __name__ == '__main__': 
    craw = Google_crawler()
    craw.run()
   