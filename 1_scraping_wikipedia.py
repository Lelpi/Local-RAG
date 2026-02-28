from dotenv import load_dotenv
import os
import json
import requests
import wikipedia

load_dotenv()


# Create datasets directory if it doesn't exist
if not os.path.exists(os.getenv("DATASET_STORAGE_FOLDER")):
    os.makedirs(os.getenv("DATASET_STORAGE_FOLDER"))

# Load keywords from keywords.json
with open("keywords.json", "r") as f:
    keywords = json.load(f)

# BrightData flow
if os.getenv("USE_BRIGHTDATA", "False") == "True":

    if os.getenv("BRIGHTDATA_API_KEY", "REPLACE") == "REPLACE":
        print("Please set your BrightData API key in the .env file before running this script or use the free version.")
        exit(1)

    headers = {
        "Authorization": f"Bearer {os.getenv('BRIGHTDATA_API_KEY')}",
        "Content-Type": "application/json",
    }

    # If no snapshot yet, trigger a new crawl and store the snapshot ID
    if not os.path.isfile(os.getenv("SNAPSHOT_STORAGE_FILE")):

        print("Starting BrightData Wikipedia scraper")

        params = {
            "dataset_id": "gd_lr9978962kkjr3nx49",
            "include_errors": "true",
            "type": "discover_new",
            "discover_by": "keyword",
        }
        json_data = [
            {"keyword": keyword, "pages_load": str(pages["brightdata"])}
            for keyword, pages in keywords.items()
        ]
        response = requests.post(
            "https://api.brightdata.com/datasets/v3/trigger",
            params=params,
            headers=headers,
            json=json_data,
        )

        result = json.loads(response.content)
        with open(os.getenv("SNAPSHOT_STORAGE_FILE"), "w") as f:
            f.write(str(result["snapshot_id"]))

        print("Scraper triggered and snapshot ID saved. Run this script again in a few minutes to fetch results.")

    # If snapshot exists, check status and fetch data if ready
    else:

        print("Checking BrightData Wikipedia scraper status")

        with open(os.getenv("SNAPSHOT_STORAGE_FILE"), "r") as f:
            snapshot_id = f.read()

        # Check the status of the crawl using the snapshot ID
        response = requests.get(
            f"https://api.brightdata.com/datasets/v3/progress/{snapshot_id}",
            headers=headers,
        )
        status = response.json()["status"]
        print(f"Status: {status}")

        if status == "ready":

            response = requests.get(
                f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
                headers=headers,
            )

            output_path = f"{os.getenv('DATASET_STORAGE_FOLDER')}/{os.getenv('DATASET_STORAGE_FILE_BRIGHTDATA')}"

            print(f"All articles are ready. Writing data to {output_path}")

            with open(output_path, "wb") as f:
                f.write(response.content)

        else:

            print("Not all articles are scraped yet. Try again in a few minutes.")

# Free (Wikipedia library) flow
else:

    wikipedia.set_lang("en")

    print("Starting free Wikipedia scraper")

    articles_data = []
    for keyword, pages in keywords.items():

        # Request more results than needed to account for disambiguation / missing pages
        search_results = wikipedia.search(keyword, results=pages["free"] * 2)
        fetched = 0
        for result in search_results:

            if fetched >= pages["free"]:
                break

            try:
                page = wikipedia.page(result, auto_suggest=False)
                print(f"Fetched: {page.title}")
                articles_data.append({
                    "url": page.url,
                    "title": page.title,
                    "raw_text": page.content,
                })
                fetched += 1

            except wikipedia.exceptions.DisambiguationError:
                print(f"Disambiguation page, skipping: {result}")
            except wikipedia.exceptions.PageError:
                print(f"Page not found: {result}")
            except Exception as e:
                print(f"Error fetching {result}: {e}")

    output_path = f"{os.getenv('DATASET_STORAGE_FOLDER')}/{os.getenv('DATASET_STORAGE_FILE_FREE')}"

    print(f"All articles fetched. Writing data to {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for article in articles_data:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
