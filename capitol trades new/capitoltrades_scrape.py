<<<<<<< HEAD
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Base URL
base_url = "https://www.capitoltrades.com/trades?page="
detail_base = "https://www.capitoltrades.com"

data = []

# Loop over a small number of pages for prototype
for page in range(1, 500):  
    print(f"Scraping page {page}...")
    url = base_url + str(page)
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to retrieve page {page}")
        continue

    soup = BeautifulSoup(response.text, 'html.parser')

    # Select all trade rows
    rows = soup.select("table tbody tr")  # More specific selector

    for row in rows:
        cells = row.find_all("td")

        # Defensive check for rows with enough cells
        if len(cells) >= 9:
            try:
                politician = cells[0].get_text(strip=True)
                issuer = cells[1].get_text(strip=True)
                published = cells[2].get_text(strip=True)
                traded = cells[3].get_text(strip=True)
                filed_after = cells[4].get_text(strip=True)
                owner = cells[5].get_text(strip=True)
                ttype = cells[6].get_text(strip=True)
                size = cells[7].get_text(strip=True)
                price = cells[8].get_text(strip=True)

                # Detail page link (in last td's <a>)
                detail_link_tag = row.select_one('a[href]')
                detail_url = detail_base + detail_link_tag["href"] if detail_link_tag else ''

                # Append cleaned row
                data.append({
                    "Politician": politician,
                    "Issuer": issuer,
                    "Published": published,
                    "Traded": traded,
                    "Filed After": filed_after,
                    "Owner": owner,
                    "Type": ttype,
                    "Size": size,
                    "Price": price,
                    "Detail URL": detail_url
                })

            except Exception as e:
                print("Error parsing row:", e)
                continue

# Save to CSV
df = pd.DataFrame(data)
df.to_csv(r"Ada-Analytics\capitol trades new\capitol_trades_data.csv", index=False)
=======
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Base URL
base_url = "https://www.capitoltrades.com/trades?page="
detail_base = "https://www.capitoltrades.com"

data = []

# Loop over a small number of pages for prototype
for page in range(1, 500):  
    print(f"Scraping page {page}...")
    url = base_url + str(page)
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to retrieve page {page}")
        continue

    soup = BeautifulSoup(response.text, 'html.parser')

    # Select all trade rows
    rows = soup.select("table tbody tr")  # More specific selector

    for row in rows:
        cells = row.find_all("td")

        # Defensive check for rows with enough cells
        if len(cells) >= 9:
            try:
                politician = cells[0].get_text(strip=True)
                issuer = cells[1].get_text(strip=True)
                published = cells[2].get_text(strip=True)
                traded = cells[3].get_text(strip=True)
                filed_after = cells[4].get_text(strip=True)
                owner = cells[5].get_text(strip=True)
                ttype = cells[6].get_text(strip=True)
                size = cells[7].get_text(strip=True)
                price = cells[8].get_text(strip=True)

                # Detail page link (in last td's <a>)
                detail_link_tag = row.select_one('a[href]')
                detail_url = detail_base + detail_link_tag["href"] if detail_link_tag else ''

                # Append cleaned row
                data.append({
                    "Politician": politician,
                    "Issuer": issuer,
                    "Published": published,
                    "Traded": traded,
                    "Filed After": filed_after,
                    "Owner": owner,
                    "Type": ttype,
                    "Size": size,
                    "Price": price,
                    "Detail URL": detail_url
                })

            except Exception as e:
                print("Error parsing row:", e)
                continue

# Save to CSV
df = pd.DataFrame(data)
df.to_csv(r"Ada-Analytics\capitol trades new\capitol_trades_data.csv", index=False)
>>>>>>> 02d397b609aa60a3641617ce607f2ae2fdcdb463
print("Scraping completed and saved.")