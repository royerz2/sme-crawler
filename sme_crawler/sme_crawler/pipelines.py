# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import pymongo
from openai import OpenAI
import os
from dotenv import load_dotenv
import time

load_dotenv()

class MongoPipeline:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def open_spider(self, spider):
        self.client = pymongo.MongoClient(os.getenv('MONGODB_URI'))
        self.db = self.client[os.getenv('MONGODB_DATABASE')]
        self.collection = self.db[os.getenv('MONGODB_COLLECTION')]
        
    def close_spider(self, spider):
        if self.client:
            self.client.close()
            
    def process_item(self, item, spider):
        # Check if article is SME-related using GPT
        is_sme_related = self._check_sme_related(item['title'], item['content'])
        
        if is_sme_related:
            # Add the classification result
            item['is_sme_related'] = True
            
            # Store in MongoDB
            self.collection.insert_one(dict(item))
            return item
        else:
            # Skip non-SME related articles
            return None
            
    def _check_sme_related(self, title, content):
        # Combine title and content for analysis
        text_to_analyze = f"Title: {title}\n\nContent: {content[:2000]}"  # Limit content length
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{
                    "role": "user",
                    "content": f"""Analyze if this text is specifically about SME (Small and Medium Enterprises) issues or problems in Limburg, Netherlands.
                    
                    Text: {text_to_analyze}
                    
                    Respond with only 'yes' if the text specifically discusses SME issues/problems in Limburg, or 'no' if it doesn't.
                    Consider it SME-related only if it explicitly mentions both SMEs and Limburg, and discusses actual issues or problems."""
                }],
                max_tokens=10,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip().lower()
            return result == 'yes'
            
        except Exception as e:
            print(f"Error in GPT analysis: {str(e)}")
            return False  # Skip article if there's an error in analysis
