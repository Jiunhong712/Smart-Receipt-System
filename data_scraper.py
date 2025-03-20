import requests
from bs4 import BeautifulSoup
import csv
import os

base_url = 'https://klec.jayagrocer.com/collections/alcohol'
start_page = 1
end_page = 14

for page in range(start_page, end_page + 1):
    url = f'{base_url}?page={page}' if page > 1 else base_url
    print(f'Scraping page {page}: {url}')

    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all product name elements
        product_elements = soup.find_all('a', class_='product-item__title text--strong link')

        # Extract product names
        product_names = [product.get_text(strip=True) for product in product_elements]

        # Print product names for this page
        if product_names:
            print(f'Found {len(product_names)} products on page {page}:')
            for name in product_names:
                print(name)
        else:
            print(f'No products found on page {page}. This might be the last page.')
            break

        # Load existing product names from CSV to avoid duplicates
        existing_names = set()
        csv_file = 'product_names (Jaya Grocer).csv'
        if os.path.exists(csv_file):
            try:
                with open(csv_file, 'r', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader, None)
                    for row in reader:
                        if row:
                            existing_names.add(row[0])
            except UnicodeDecodeError:
                with open(csv_file, 'r', encoding='latin1') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader, None)
                    for row in reader:
                        if row:
                            existing_names.add(row[0])
                print("Warning: File was read with 'latin1' encoding due to UTF-8 decode error.")

        # Append new product names to CSV
        new_count = 0
        with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
                writer.writerow(['Product Name'])
            # Add only new product names
            for name in product_names:
                if name not in existing_names:
                    writer.writerow([name])
                    existing_names.add(name)
                    new_count += 1

        print(f'Added {new_count} new product names from page {page}. Total unique names: {len(existing_names)}')
    else:
        print(f'Failed to retrieve page {page}. Status code: {response.status_code}')
        break

print(f'Scraping complete. Check {csv_file} for all product names.')