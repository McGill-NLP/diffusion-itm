import pandas as pd
import requests
import json
import os

def is_downloadable(url):
    """
    Checks if a given url is downloadable by making a HEAD request
    """
    print(f"Checking URL: {url}")
    try:
        h = requests.head(url, allow_redirects=True)
        header = h.headers
        content_type = header.get('content-type')
        if 'text' in content_type.lower():
            print(f"URL is not an image file: {url}")
            return False
        if 'html' in content_type.lower():
            print(f"URL is an HTML file, not image: {url}")
            return False
        print(f"URL is valid: {url}")
        return True
    except:
        print(f"URL checking failed: {url}")
        return False

def download_image(url, file_name):
    """
    Downloads an image from a given url and saves it with a given file name
    """
    print(f"Downloading image from URL: {url}")
    response = requests.get(url, stream=True)
    file = open(file_name, 'wb')
    response.raw.decode_content = True
    file.write(response.raw.read())
    file.close()
    print(f"Image saved as: {file_name}")

def main():
    # Load the csv
    print("Loading CSV file...")
    data = pd.read_csv('svo_probes.csv')
    print("CSV file loaded successfully.")
    
    # Limiting to first 10 rows for testing
    # data = data.head(10)

    # Initialize an empty list to hold valid entries
    json_data = []

    # Image directory
    image_dir = 'images/'
    if not os.path.exists(image_dir):
        print(f"Creating directory: {image_dir}")
        os.makedirs(image_dir)

    for index, row in data.iterrows():
        print(f"Processing row {index + 1}...")
        pos_url = row['pos_url']
        neg_url = row['neg_url']
        if is_downloadable(pos_url) and is_downloadable(neg_url):
            # Download and save images
            pos_image_id = str(row['pos_image_id']) + '.jpg'
            neg_image_id = str(row['neg_image_id']) + '.jpg'
            download_image(pos_url, image_dir + pos_image_id)
            download_image(neg_url, image_dir + neg_image_id)
            
            # Determine neg_type
            neg_type = "subj" if row['subj_neg'] else ("verb" if row['verb_neg'] else "obj")
            
            # Append a new dictionary to our list
            json_data.append({
                "pos_id": pos_image_id,
                "neg_id": neg_image_id,
                "sentence": row['sentence'],
                "pos_triplet": row['pos_triplet'],
                "neg_triplet": row['neg_triplet'],
                "neg_type": neg_type
            })
            print(f"Row {index + 1} processed successfully.")
        else:
            print(f"Skipping row {index + 1} due to invalid URLs.")

        # Write out the json file
        print("Writing data to JSON file...")
        with open('svo.json', 'w') as outfile:
            json.dump(json_data, outfile, indent=4)
        print("JSON file created successfully.")

if __name__ == '__main__':
    main()
