import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import os

#working directories
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
DATA_DIR = os.path.join(BASE_DIR, 'get_data','data')

#beautiful soup and selenium objects
driver = webdriver.Chrome(executable_path='C:/Program Files/selenium/chromedriver.exe')

#range of seasons wished to scrape
seasons_wished = list(reversed(range(2005,2020)))

#complement of all urls
rest_of_url = '/results/'

root_url = 'https://www.oddsportal.com/soccer/england/premier-league'

#first url
main_url = root_url + rest_of_url

#all other season url
seasons_url = [root_url + '-' + str(season) + '-' + str(season + 1) + rest_of_url for season in seasons_wished]

#complete url list to be scraped
all_urls = [main_url] + seasons_url
all_urls

#dataframe that will be populated
df = pd.DataFrame()

#function to check if variable is null
def is_empty(col):
    try:
        result = col.text
    except:
        result = None
    return result

#scraping
for url in all_urls:

    #selenium and soup objects
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser', from_encoding="utf-8")

    while True:

        #store previous or current page number
        previous_page = soup.find_all( 'span', attrs = {'class' : 'active-page'})[0].text

        for col in soup.find_all('tr', attrs = {'deactivate'}):
            df = df.append(
            {
            #match season
            'season' : soup.find_all('span', attrs = {'class' : 'active'})[1].text,
            
            #match date
            'date' : col.findPreviousSibling(attrs = {'center nob-border'}).text[0:-6],

            #match name
            'match_name' : col.find('td', attrs = {'class' : 'name table-participant'}).text.replace('\xa0', ''),

            #match result
            'result' : col.find('td', attrs = {'class' : 'center bold table-odds table-score'}).text,

            #home winning odd
            'h_odd' : is_empty(col.find('td', attrs = {'class' : "result-ok odds-nowrp"})),

            #draw odd
            'd_odd' : is_empty(col.find('td', attrs = {'class' : "odds-nowrp"}).findNext( attrs = {'class' : "odds-nowrp"})),

            #away winning odd
            'a_odd' : is_empty(col.find('td', attrs = {'class' : "odds-nowrp"}).findNext( attrs = {'class' : "odds-nowrp"}).findNext( attrs = {'class' : "odds-nowrp"}))
            }
            , ignore_index = True)

        print('page done!')

        #clicks on next page
        element = driver.find_element_by_partial_link_text('»')
        driver.execute_script("arguments[0].click();", element)

        #sleep so that the page can load properly
        time.sleep(2)

        #reload soup objects on new page
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser', from_encoding="utf-8")

        #get new page number
        new_page = soup.find_all('span', attrs = {'class' : 'active-page'})[0].text

        #if there's no new pages left break
        if previous_page != new_page:
            continue
        else:
            break

    print(url, 'done!')

driver.quit()
print('scraping finished!')

#saving full df to csv
df.to_csv(os.path.join(DATA_DIR,  'matches.csv'), index = False)