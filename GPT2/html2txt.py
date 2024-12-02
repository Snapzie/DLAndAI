import requests
import os
from bs4 import BeautifulSoup

# Function to scrape text from a webpage
def scrape_text(url):
    # Fetch the content from the url
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get the text and remove extra whitespace
        text = soup.get_text(separator='\n', strip=True)
        return text
    else:
        return f"Failed to retrieve content. Status code: {response.status_code}"

# URL of the webpage to scrape
url = 'https://en.wikipedia.org/wiki/Atom'

# Scrape and print the text from the webpage
text = scrape_text(url)

with open('./html2txt.txt','w') as f:
    f.write(text)

