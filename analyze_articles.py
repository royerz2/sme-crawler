import pymongo
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
import json
from bson import ObjectId
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# MongoDB connection details for articles
MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION')

# MongoDB connection details for analysis results
MONGODB_ANALYSIS_DATABASE = os.getenv('MONGODB_ANALYSIS_DATABASE')
MONGODB_ANALYSIS_COLLECTION = os.getenv('MONGODB_ANALYSIS_COLLECTION')

# OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()
client_openai = OpenAI(api_key=OPENAI_API_KEY)

# Connect to MongoDB
try:
    client = pymongo.MongoClient(MONGODB_URI)
    db = client[MONGODB_DATABASE]
    articles_collection = db[MONGODB_COLLECTION]
    analysis_db = client[MONGODB_ANALYSIS_DATABASE]
    analysis_collection = analysis_db[MONGODB_ANALYSIS_COLLECTION]
    print("Connected to MongoDB.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

# SME-related keywords for prompt (ensure this matches your api.py)
SME_PROBLEM_KEYWORDS = {
    'financing': ['krediet', 'lening', 'financiering', 'cash', 'liquiditeit', 'kapitaal', 'investering', 'budget'],
    'workforce': ['personeel', 'medewerkers', 'werknemers', 'recruitment', 'arbeidskrachten', 'talententekort'],
    'regulation': ['regelgeving', 'wet', 'belasting', 'compliance', 'bureaucratie', 'vergunning'],
    'technology': ['digitalisering', 'technologie', 'automatisering', 'it', 'software', 'cyber'],
    'market_competition': ['concurrentie', 'markt', 'klanten', 'verkoop', 'omzet', 'prijzen'],
    'supply_chain': ['leveranciers', 'toeleveringsketen', 'logistiek', 'transport', 'voorraad'],
    'sustainability': ['duurzaamheid', 'milieu', 'klimaat', 'energie', 'co2', 'groen'],
    'innovation': ['innovatie', 'onderzoek', 'ontwikkeling', 'nieuwe', 'vernieuwing']
}

def analyze_article_with_gpt(article):
    """Performs GPT-4.1 Nano sentiment analysis on a single article."""
    combined_text = (article.get('title', '') + ' ' + article.get('content', '')).strip()
    if not combined_text:
        return {"error": "No text to analyze"}

    # Define the prompt for GPT-4.1 Nano
    prompt = f"""Analyze the following news article about SMEs and provide the following:
1. Overall Sentiment Score (-1 to +1)
2. Sentiment towards Key Entities/Topics (e.g., {list(SME_PROBLEM_KEYWORDS.keys())})
3. Emotional Tone (e.g., informative, critical, optimistic)
4. Actionability/Implication for SMEs (brief summary)
Provide the response in JSON format with keys: overall_score, entity_sentiment, emotional_tone, implications. Ensure entity_sentiment is a dictionary where keys are entities and values are sentiment scores or descriptions.

Article Title: {article.get('title', '')}
Article Content: {combined_text[:2000]}...""" # Limit content length for prompt

    try:
        response = client_openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes news articles about SMEs and provides structured sentiment analysis in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        analysis_text = response.choices[0].message.content.strip()

        try:
            advanced_analysis = json.loads(analysis_text)
            return advanced_analysis
        except json.JSONDecodeError as json_e:
            print(f"Failed to parse GPT response JSON for article {article.get('_id', 'N/A')}: {json_e}")
            return {"error": "Failed to parse GPT response", "raw_response": analysis_text}

    except Exception as gpt_e:
        print(f"GPT analysis failed for article {article.get('_id', 'N/A')}: {gpt_e}")
        return {"error": f"GPT analysis failed: {gpt_e}"}

def analyze_all_articles():
    """Iterates through articles, performs analysis if missing, and stores results."""
    print("Starting analysis of existing articles...")
    
    articles_to_analyze_cursor = articles_collection.find({})
    total_articles = articles_collection.count_documents({})
    analyzed_count = 0
    skipped_count = 0

    for article in articles_to_analyze_cursor:
        article_id_str = str(article['_id'])

        # Check if analysis already exists for this article ID
        existing_analysis = analysis_collection.find_one({'article_id': article_id_str})
        
        if existing_analysis:
            print(f"Analysis already exists for article {article_id_str}. Skipping.")
            skipped_count += 1
            continue

        print(f"Analyzing article: {article_id_str}")
        analysis_results = analyze_article_with_gpt(article)

        # Store the analysis results with a reference to the article ID
        analysis_doc = {
            'article_id': article_id_str,
            'analysis': analysis_results,
            'analyzed_at': datetime.utcnow().isoformat()
        }

        try:
            analysis_collection.insert_one(analysis_doc)
            print(f"Analysis saved for article: {article_id_str}")
            analyzed_count += 1
        except Exception as e:
            print(f"Failed to save analysis for article {article_id_str}: {e}")

        # Optional: Add a small delay to avoid hitting API rate limits
        time.sleep(1) # Adjust delay as needed

    print("\nAnalysis complete.")
    print(f"Total articles processed: {total_articles}")
    print(f"Articles analyzed and saved: {analyzed_count}")
    print(f"Articles skipped (analysis already exists): {skipped_count}")

if __name__ == "__main__":
    analyze_all_articles()
    client.close()