import os
import requests
from urllib.parse import urlparse, parse_qs

# Read URLs from file
with open("links.txt", "r") as f:
    urls = f.read().splitlines()

    # Ensure data folder exists
    os.makedirs("data", exist_ok=True)

# Download each file
for url in urls:
    filename = os.path.basename(url)
    try:
        print(f"Downloading {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            filename = qs.get("LABEL", [os.path.basename(parsed.path)])[0]
            filename = filename.replace("/", "_")  # Make filename safe
            filepath = os.path.join("data", filename)
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded: {filepath}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")