import openai
from openai import OpenAI
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Basic text classification with GPT-4.1 Nano
text_to_classify = "This product is amazing! I love it so much."
categories = ["positive", "negative", "neutral"]

start_time = time.time()

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[{
        "role": "user", 
        "content": f"Classify this text into one of these categories: {', '.join(categories)}\n\nText: {text_to_classify}\n\nRespond with only the category name."
    }],
    max_tokens=10,
    temperature=0.1
)

end_time = time.time()

print(f"Classification: {response.choices[0].message.content}")
