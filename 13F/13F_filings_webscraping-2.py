#!/usr/bin/env python
# coding: utf-8

# In[12]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta


# In[14]:


df = pd.DataFrame({
    'institution_name': [
    "Berkshire Hathaway",
    "Renaissance Technologies",
    "Citadel Advisors",
    #"BlackRock Inc.",
    "Bridgewater Associates",
    #"Two Sigma Investments",
    #"Millennium Management",
    #"Vanguard Group",
    #"FMR LLC",
    "State Street Corp"
],
    'cik': [
    "0001067983",
    "0001166559",
    "0001037389",
    #"0001081060",
    "0001103804",
    #"0000922971",
    #"0001079114",
    #"0000912057",
    #"0000316927",
    "0000354204" 
]
})
df['cik'] = df['cik'].astype(str).str.zfill(10)
df.to_csv('top_institutions.csv', index=False)


# In[16]:


institutions_df = pd.read_csv('top_institutions.csv', dtype={'cik': str})
institutions_df['cik'] = institutions_df['cik'].str.zfill(10)

top_institutions = dict(zip(institutions_df['cik'], institutions_df['institution_name']))


# since we only want recent filings and for the coparision, we need two of the recent filings of any institution, we took 180 days in the function below, because, the institutions file one fpor every three months ~ 90 days. and there is a buffer of 45 days before the deadline. so it is 135 days approx. and for a safety buffer, we round it upto 180 days, which is exactly two quarters. 

# In[18]:


# with date filter, just to get the recent filings. 

def fetch_filing_metadata(cik, institution_name, days_ago=180):
    cutoff_date = datetime.utcnow() - timedelta(days=days_ago)

    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=13F-HR&owner=exclude&count=40&output=atom"
    headers = {
        "User-Agent": "DataScienceInternshipBot/1.0 (contact: chandanarchutha.n@gmail.com)",
        "Accept": "application/xml",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.sec.gov/"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "xml")
    entries = soup.find_all("entry")

    filing_links = []
    filed_dates = []
    institution_names = []

    for entry in entries:
        date_str = entry.find("updated").text.strip()
        date_obj = datetime.strptime(date_str[:10], "%Y-%m-%d")  # keep only YYYY-MM-DD

        if date_obj >= cutoff_date:
            link = entry.find("link")["href"].strip()
            filing_links.append(link)
            filed_dates.append(date_str)
            institution_names.append(institution_name)

    return filing_links, filed_dates, institution_names


# In[20]:


# just to make sure we got recent two filings:

for cik, institution_name in top_institutions.items():
    filing_links, filed_dates, institution_names = fetch_filing_metadata(cik, institution_name, days_ago=180)

    # Convert filed_dates to datetime and sort
    filed_dates_cleaned = sorted([pd.to_datetime(date[:10]) for date in filed_dates], reverse=True)

    print(f"\n📄 {institution_name} ({cik})")
    for i, d in enumerate(filed_dates_cleaned[:2]):
        print(f"   Filing {i+1}: {d.date()}")


# **Extract XML links**

# In[22]:


# Function to extract XML info table link from index page
def extract_info_table_xml(index_url):
    headers = {
        "User-Agent": "DataScienceInternshipBot/1.0 (contact: chandanarchutha.n@gmail.com)",
        "Accept-Language": "en-US,en;q=0.9"
    }

    try:
        response = requests.get(index_url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table", class_="tableFile")
        if table:
            for row in table.find_all("tr"):
                cols = row.find_all("td")
                if len(cols) >= 3:
                    a_tag = cols[2].find("a")
                    doc_type = cols[3].text.strip().lower()
                    if a_tag:
                        href = a_tag.get("href", "")
                        if href.endswith(".xml") and "form13f" in href.lower() and doc_type == "information table":
                            return "https://www.sec.gov" + href
    except Exception as e:
        print(f"Error parsing {index_url}: {e}")
    return None


# **Extract the info table from the xml links**

# In[24]:


def extract_13f_holdings_from_html(xml_url, filed_date, institution_name):
    headers = {
        "User-Agent": "DataScienceInternshipBot/1.0 (contact: chandanarchutha.n@gmail.com)",
        "Accept": "application/xml",
        "Accept-Language": "en-US,en;q=0.9"
    }

    response = requests.get(xml_url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    tables = soup.find_all("table")
    if not tables:
        return []

    data_table = tables[-1]
    rows = data_table.find_all("tr")
    holdings = []

    for row in rows[4:]:
        cols = row.find_all("td")
        if len(cols) >= 13:
            issuer = cols[0].text.strip()
            class_title = cols[1].text.strip()
            cusip = cols[2].text.strip()
            value = cols[4].text.strip()
            shares = cols[5].text.strip()
            discretion = cols[8].text.strip()
            voting_sole = cols[10].text.strip()
            voting_shared = cols[11].text.strip()
            voting_none = cols[12].text.strip()

            holdings.append([
                issuer, class_title, cusip, value, shares, discretion,
                voting_sole, voting_shared, voting_none, filed_date, xml_url, institution_name
            ])
    return holdings


# **Fetching the data at once for everything**

# In[26]:


# with time thing. there is a pause between each request

import time

# Final version of scraping loop with improvements
all_holdings = []
failed_urls = []

for cik, institution_name in top_institutions.items():
    filing_links, filed_dates, institution_names = fetch_filing_metadata(cik, institution_name)
    xml_links = [extract_info_table_xml(link) for link in filing_links]
    sample_filings = list(zip(xml_links, filed_dates, institution_names))
    for i, (xml_url, filed_date, institution_name) in enumerate(sample_filings):
        if xml_url:
            try:
                extracted = extract_13f_holdings_from_html(xml_url, filed_date, institution_name)
                all_holdings.extend(extracted)

                time.sleep(0.5)  # polite pause between requests to avoid SEC throttling
            except Exception as e:
                print(f"Error parsing {xml_url}: {e}")
                failed_urls.append((xml_url, institution_name))




if failed_urls:
    print(f"\n❌ {len(failed_urls)} filings failed to parse. You can retry them later if needed.")
    for url, name in failed_urls:
        print(f"- {name}: {url}")



# In[28]:


data = pd.DataFrame(all_holdings, columns=[
    "Issuer", "Class", "CUSIP", "Value (x$1000)", "Shares", "Discretion",
    "Voting - Sole", "Voting - Shared", "Voting - None", "Filed Date", "Source URL", "Institution"
])        


# In[ ]:


# data.to_csv('13F_filings.csv', index=False)


# In[30]:


data.info()


# In[32]:


data.head(10)


# Even though all 5 rows are about ALLY FINL INC, each one refers to:
# 
# - A different block of shares
# - Different voting authority configurations
# - Potentially different internal managers at Berkshire handling that position
# 

# ## mapping the data with the ticker symbol file based on cusip

# In[34]:


data['Shares'] = data['Shares'].astype(str).str.replace(',', '').astype(int)
data['Filed Date'] = pd.to_datetime(data['Filed Date'], utc=True)
data['CUSIP'] = data['CUSIP'].astype(str).str.strip().str.upper()


# In[36]:


import requests
import pandas as pd
import time

# Your OpenFIGI API key
API_KEY = '8dd35fce-dce2-476f-9af5-e6abdbd452e4'

headers = {
    'Content-Type': 'text/json',
    'X-OPENFIGI-APIKEY': API_KEY
}

def get_tickers_from_cusips(cusips):
    cusips = list(set(cusips))  # remove duplicates
    results = []

    for i in range(0, len(cusips), 100):
        batch = cusips[i:i+100]
        query = [{"idType": "ID_CUSIP", "idValue": cusip} for cusip in batch]

        response = requests.post('https://api.openfigi.com/v3/mapping', headers=headers, json=query)

        if response.status_code != 200:
            print(f"Error with request: {response.status_code}")
            continue

        data = response.json()

        for item, cusip in zip(data, batch):
            if item and 'data' in item and item['data']:
                ticker = item['data'][0].get('ticker')
            else:
                ticker = None
            results.append({'CUSIP': cusip, 'Ticker': ticker})

        time.sleep(1)  # Be polite to the API

    return pd.DataFrame(results)


# In[38]:


unique_cusips = data['CUSIP'].dropna().unique().tolist()
cusip_ticker_df = get_tickers_from_cusips(unique_cusips)


# In[40]:


data = data.merge(cusip_ticker_df, on='CUSIP', how='left')


# In[42]:


data.head()


# ## Tracking poition changes

# For each institution in dataset, we’ll:
# - Filter all their filings
# - Sort by Filed Date
# - Get the latest two distinct filing dates
# 
# 
# 

# In[44]:


# Build latest_filing_dates dictionary from your clean DataFrame
latest_filing_dates = {}

for inst in data['Institution'].unique():
    inst_dates = data[data['Institution'] == inst]['Filed Date'].drop_duplicates().sort_values(ascending=False)

    if len(inst_dates) >= 2:
        latest_filing_dates[inst] = {
            'latest': inst_dates.iloc[0],
            'previous': inst_dates.iloc[1]
        }
    else:
        print(f"⚠️ Skipping {inst}: Less than 2 filings found.")


# In[46]:


position_changes = []  # store all outputs

for inst_name, dates in latest_filing_dates.items():
    latest_date = dates['latest']
    prev_date = dates['previous']

    df_latest = data[(data['Institution'] == inst_name) & (data['Filed Date'] == latest_date)]
    df_prev = data[(data['Institution'] == inst_name) & (data['Filed Date'] == prev_date)]

    merged = df_latest.merge(
        df_prev,
        on="CUSIP",
        how="outer",
        suffixes=("_latest", "_prev")
    )

    # ✅ Handle Ticker: take from latest, or fallback to previous
    merged['Ticker'] = merged.get('Ticker_latest', None).combine_first(merged.get('Ticker_prev', None))
    merged.drop(columns=['Ticker_latest', 'Ticker_prev'], inplace=True, errors='ignore')

    # Clean Shares columns
    merged['Shares_latest'] = pd.to_numeric(merged['Shares_latest'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
    merged['Shares_prev'] = pd.to_numeric(merged['Shares_prev'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)

    # Add Institution and Filing Dates for context
    merged['Institution'] = inst_name
    merged['Latest Filed Date'] = latest_date
    merged['Previous Filed Date'] = prev_date

    # Classification function
    def classify_change(row):
        if row['Shares_prev'] == 0 and row['Shares_latest'] > 0:
            return 'new'
        elif row['Shares_latest'] == 0 and row['Shares_prev'] > 0:
            return 'exited'
        elif row['Shares_latest'] > row['Shares_prev']:
            return 'increase'
        elif row['Shares_latest'] < row['Shares_prev']:
            return 'decrease'
        else:
            return 'unchanged'

    merged['Change Type'] = merged.apply(classify_change, axis=1)

    # Append only those that exist in the latest filing (i.e. still held or newly bought)
    current_holdings = merged[merged['Shares_latest'] > 0]
    position_changes.append(current_holdings)

# Combine all institutions' changes into one DataFrame
final_changes_df = pd.concat(position_changes, ignore_index=True)


# In[48]:


final_changes_df.info()


# In[50]:


final_changes_df.head()


# In[52]:


print(final_changes_df['Change Type'].value_counts())


# In[54]:


print(final_changes_df[['Shares_latest', 'Shares_prev', 'Change Type']].head(10))



# In[ ]:




