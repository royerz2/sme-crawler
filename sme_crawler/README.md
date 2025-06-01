# SME News Crawler for Limburg

This Scrapy crawler collects news articles about SMEs in Limburg from various sources and stores them in MongoDB. It uses GPT to filter articles and only keeps those that specifically discuss SME issues in Limburg.

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with the following variables:
```
MONGODB_URI=your_mongodb_connection_string
MONGODB_DATABASE=your_database_name
MONGODB_COLLECTION=your_collection_name
OPENAI_API_KEY=your_openai_api_key
```

## Usage

To start the crawler:
```bash
cd sme_crawler
scrapy crawl sme_spider
```

The crawler will:
1. Visit the specified websites
2. Extract article content
3. Use GPT to analyze if the content is specifically about SME issues in Limburg
4. Store only the relevant articles in MongoDB

## Features

- Crawls multiple news websites
- Extracts article title, content, and metadata
- Uses GPT to filter for SME-specific content in Limburg
- Stores filtered articles in MongoDB
- Respects robots.txt and implements polite crawling
- Implements caching to avoid unnecessary requests

## Websites Crawled

- ondernemeninlimburg.nl
- sittard-geleen.nieuws.nl
- mkblimburg.nl
- brightlands.com
- pomlimburg.be
- business.gov.nl 