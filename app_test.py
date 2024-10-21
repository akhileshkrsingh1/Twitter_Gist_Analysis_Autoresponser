from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from translate import Translator as TranslateLib
from ai4bharat.transliteration import XlitEngine
from langdetect import detect
from pymongo import MongoClient
from docx import Document
import time
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.secret_key = os.urandom(26)

with open('data/crime_categories.txt', 'r', encoding='utf-8') as file:
    crime_categories = file.read()
with open('data/priority_list.txt', 'r', encoding='utf-8') as file:
    priority_list = file.read()
with open('data/crime_list.json', 'r') as f:
    crime_data = json.load(f)
    crime_list = [item['crime_type'].lower() for item in crime_data['crime']]


client = MongoClient("mongodb+srv://dummyuser:dummy@cluster0.qvmspwg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["tweet_analysis"]
tweets_collection = db["tweets"]
drafts_collection = db["tweet_drafts"]
counters_collection = db["counters"]

def load_responses(doc_path):
    doc = Document(doc_path)
    response_dict = {}
    
    for table in doc.tables:
        for row in table.rows:
            cells = row.cells
            if len(cells) == 3:
                category = cells[1].text.strip().lower()
                response = cells[2].text.strip()
                response_dict[category] = response
    
    return response_dict

def get_responses_for_categories(categories, response_dict):
    responses = {}
    for category in categories:
        if category in response_dict:
            responses[category] = response_dict[category]
    return responses


memory = ConversationBufferWindowMemory(k=0, memory_key="chat_history", return_messages=True)
response_dict = load_responses('data/response_list.docx')

def load_embeddings():
    """Load and cache the embeddings model."""
    return HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

embeddings = load_embeddings()
db = FAISS.load_local("ipc_embed_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

db1 = FAISS.load_local("categories_db", embeddings, allow_dangerous_deserialization=True)
db_retriever1 = db1.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define prompt templates for each task
sentiment_prompt_template = f"""
<s>[INST]
As an expert sentiment analyzer,
you are tasked with providing highly
accurate sentiment analysis.
Ensure your answers
meet these criteria:
Respond in a bullet-point format
and in less than 5 words
Analyze the sentiment of this text based on the user who has writen the text in the format (sentiment:).
Also their are only three sentiments neutral, positive, negative. 
CONTEXT: {{context}}
QUESTION: {{question}}
</s>[INST]
"""


category_prompt_template = f"""
<s>[INST]
"As an expert in categorizing crimes according to the Indian Penal Code, sentiment analysis, and gist extraction, your task is to provide the top 3 most relevant crime categories based on the given tweet. Follow these steps:
Analyze the gist of the tweet.
Compare the tweet's content with the definitions in the {crime_categories} reference file containing the crime categories.
Perform similarity analysis on both the tweet and the relevant definitions.
If the similarity between the tweet and a definition is approximately 80% or higher, consider it a match and also don't make any assumptions about the tweet.If anything is not explicitly mentioned strictly do no consider it in analysis. Also don't consider not explicitly mentioned in the tweet.
Return up to 3 most relevant crime categories, each in 1-3 words.

If fewer than 3 categories match, only list the matching ones. Ensure your responses are professional, authoritative, and informative for Delhi Police use.

CONTEXT: {{context}}
QUESTION: {{question}}
</s>[INST]
"""

response_prompt_template = f"""
<s>[INST]
You are an AI assistant tasked with generating Twitter responses for the Delhi Police's official account. Your responses should be:
Concise and within Twitter's character limit of 200 words
Professional and authoritative in tone
Informative and helpful to citizens
Sensitive to public safety concerns
In line with Delhi Police's official communication style
Out of these responses first response should be high priority according to the gist sentiment 
Second should be that is common in all gist , third response should suggest some special references or helpline numbers and other referencing authorities and fourth response should contain some special suggestion to the victim or user.
This is the tweet
CONTEXT: {{context}}
QUESTION: {{question}}

Refer to the following responses for the crime categories and generate four responses to the tweet.

"""

priority_prompt_template = f"""
<s>[INST]
You are an AI assistant tasked with analyzing tweets and assigning priorities based on their sentiment. A tweet can have either a "High Priority" or "Low Priority". If the tweet requires immediate attention (e.g., it involves a serious complaint, urgent issue, or critical situation also refer to this list for assigning priority {priority_list}), assign it "High Priority". Otherwise, assign it "Low Priority".
This is the tweet
CONTEXT:{{context}}
QUESTION:{{question}}
"""

# Create PromptTemplate instances for each task
sentiment_prompt = PromptTemplate(template=sentiment_prompt_template, input_variables=['context', 'question'])
category_prompt = PromptTemplate(template=category_prompt_template, input_variables=['context', 'question'])
# response_prompt = PromptTemplate(template=response_prompt_template, input_variables=['context', 'question'])

chat = ChatGroq(model_name="llama3-8b-8192", temperature=0.2, groq_api_key="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")
chat1 = ChatGroq(model_name="llama3-8b-8192", temperature=0.1, groq_api_key="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")


# Define the individual chains
sentiment_chain = ConversationalRetrievalChain.from_llm(
    llm=chat, 
    memory=memory, 
    retriever=db_retriever, 
    combine_docs_chain_kwargs={'prompt': sentiment_prompt}
)

category_chain = ConversationalRetrievalChain.from_llm(
    llm=chat1, 
    memory=memory, 
    retriever=db_retriever1, 
    combine_docs_chain_kwargs={'prompt': category_prompt}
)


# Define the parallel chains
parallel_chain = RunnableParallel(branches={
    "sentiment": sentiment_chain,
    "categories": category_chain
    # "response": response_chain
})

def response_chain_define(updated_prompt):
    response_chain = ConversationalRetrievalChain.from_llm(
    llm=chat, 
    memory=memory, 
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': updated_prompt}
)
    return response_chain

def priority_chain_define(updated_prompt):
    priority_chain = ConversationalRetrievalChain.from_llm(
    llm=chat, 
    memory=memory, 
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': updated_prompt}
)
    return priority_chain

def extract_categories(categories_answer):
    """Extract categories from the model response."""
    # Match categories that start with a bullet point or that appear at the beginning of a line
    categories_list = re.findall(r'\*\*(.*?)\*\*', categories_answer)
    # Strip any extra whitespace
    categories = [cat.strip() for cat in categories_list]
    return categories

def detect_language_and_translate(text):
    """Detects the language of the text and translates to English if necessary."""
    detected_lang = detect(text)
    # translation = TranslateLib(to_lang="en")
    translated_text = ""
    print(f"Detected language: {detected_lang}")
    if detected_lang == 'en':
        return text, 'en'
    
    elif detected_lang == 'hi':
        print("yes coming here")
        translated_text = hindi_to_english(text)
    elif is_hinglish(text):
        transliterated_text = transliterate_hinglish_to_hindi(text)
        translated_text = hindi_to_english(transliterated_text)
    print(f"Translated text: {translated_text}")
    return translated_text, detected_lang

def translate_to_hindi(text):
    """Translates text to Hindi."""
    translator = TranslateLib(to_lang="hi")
    return translator.translate(text)

def is_hinglish(text):
    """Detects if the text is in Hinglish."""
    hindi_pattern = re.compile(r'[\u0900-\u097F]')
    english_pattern = re.compile(r'[a-zA-Z]')
    
    has_hindi = bool(hindi_pattern.search(text))
    has_english = bool(english_pattern.search(text))
    
    return not has_hindi and has_english

def hindi_to_english(text):
    translator = TranslateLib(to_lang="en", from_lang="hi")
    translation = translator.translate(text)
    return translation

def transliterate_hinglish_to_hindi(text):
    """Transliterates Hinglish text to Hindi."""
    engine = XlitEngine("hi", beam_width=10, rescore=True)
    transliterated_words = []
    words = text.split()
    for word in words:
        transliterated_word = engine.translit_word(word, topk=1)
        transliterated_words.append(transliterated_word['hi'][0])
    return ' '.join(transliterated_words)

def preprocess_text(text):
    """Preprocesses the input text by removing links, @username mentions, and ensuring @DelhiPolice is added."""
    text = re.sub(r'http\S+', '', text)  # Remove links
    text = re.sub(r'@\w+', '', text)  # Remove @username mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    if not text.startswith('@DelhiPolice'):
        text = '@DelhiPolice ' + text  # Ensure @DelhiPolice is added in front
    return text.strip()

def determine_profile(followers, retweets,likes=0):
    try:
        if followers is None:
            followers=0
        else:
            followers = int(followers)
        if retweets is None:
            retweets=0
        else:
            retweets = int(retweets)
    except ValueError:
        followers = 0
        retweets = 0
        
    if followers > 10000 or retweets > 1000 or likes > 1000:
        return 'High Engagement'
    else:
        return 'Low Engagement'

def get_next_sequence_value(sequence_name):
    """Get the next value for a sequence (auto-increment)."""
    counter = counters_collection.find_one_and_update(
        {"_id": sequence_name},
        {"$inc": {"sequence_value": 1}},
        return_document=True,
        upsert=True
    )
    return counter["sequence_value"]

def extract_responses(response_answer,username=""):
    # Use a regex pattern to match each heading and capture the text after it
    response_pattern = r'\*\*(.*?)\*\*(.*?)(?=\*\*|$)'
    matches = re.findall(response_pattern, response_answer, re.DOTALL)
    
    # Return the extracted responses as a list of dictionaries
    if username!="" and username is not None:
        return [{'heading': match[0].strip(), 'text': f'@{username} ' + match[1].strip()} for match in matches]
    
    return [{'heading': match[0].strip(), 'text':match[1].strip()} for match in matches]

# Extracting the username from the tweet data
def get_username_from_tweet(tweet_text):
    username_match = re.search(r'@(\w+)', tweet_text)
    return username_match.group(1) if username_match else ""

# Post processing pipeline for categories
def filter_categories(categories):
    return [category for category in categories if category.lower() in crime_list]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tweets', methods=['GET'])
def get_tweets():
    try:
        tweets = list(tweets_collection.find())
        for tweet in tweets:
            tweet['_id'] = str(tweet['_id'])  # Convert ObjectId to string for JSON serialization
            tweet['profile'] = determine_profile(tweet.get('followers', 0), tweet.get('retweets', 0))
            tweet['tweet_id'] = int(tweet['tweet_id'])  # Ensure tweet_id is an integer
        tweets.sort(key=lambda x: x['tweet_id'], reverse=True)  # Sort tweets by tweet_id in descending order
        return jsonify({"tweets": tweets})
    except Exception as e:
        logger.error(f"Error retrieving tweets: {e}")
        return jsonify({"error": "An error occurred while retrieving tweets"}), 500

@app.route('/draft/<tweet_id>', methods=['GET'])
def get_draft(tweet_id):
    try:
        # Ensure tweet_id is an integer if necessary
        try:
            tweet_id = int(tweet_id)
        except ValueError:
            return jsonify({"error": "Invalid tweet ID"}), 400

        draft = drafts_collection.find_one({"id": tweet_id})
        if draft:
            draft['_id'] = str(draft['_id'])  # Convert ObjectId to string for JSON serialization
            return jsonify(draft)
        return jsonify({"error": "Draft not found"}), 404
    except Exception as e:
        logger.error(f"Error retrieving draft {tweet_id}: {e}")
        return jsonify({"error": "An error occurred while retrieving the draft"}), 500

@app.route('/upload', methods=['POST'])
def upload_tweets():
    try:
        tweets = request.json  # Directly get the list of tweets from the JSON payload
        for tweet in tweets:
            tweet_data = {
                "tweet_id": get_next_sequence_value("tweet_id"),  # Use auto-incremented value
                "text": tweet.get("tweet"),
                "url": tweet.get("tweet_url"),
                "username": tweet.get("twitter_account").split("\n")[0],
                "image": tweet.get("image_url"),
                "retweets": tweet.get("Number_of_retweet"),
                "followers": tweet.get("number_of_followers")
            }
            
            existing_tweet = tweets_collection.find_one({"tweet_id": tweet_data["tweet_id"]})
            if not existing_tweet:
                retries = 3
                while retries > 0:
                    try:
                        tweets_collection.insert_one(tweet_data)
                        # Analyze and save the response in drafts_collection
                        if tweet_data['text']:
                            username = tweet_data['username']
                            username = get_username_from_tweet(tweet_data['username'])
                            preprocessed_data = preprocess_text(tweet_data['text'])
                            translated_prompt, original_language = detect_language_and_translate(preprocessed_data)
                            print(f"Translated prompt: {translated_prompt} (original language: {original_language})")

                            print("Invoking parallel chains")
                            result = parallel_chain.invoke({"question": translated_prompt})
                            print(f"Parallel chain result: {result}")

                            # Extract sentiment
                            sentiment_answer = result.get("branches", {}).get("sentiment", {}).get("answer", "")
                            sentiment_match = re.search(r'Sentiment:\s*(\w+)', sentiment_answer, re.IGNORECASE)
                            sentiment = sentiment_match.group(1).strip() if sentiment_match else "unknown"

                            categories = []
                            if sentiment.lower() == "negative":
                                # Extract categories
                                categories_answer = result.get("branches", {}).get("categories", {}).get("answer", "")
                                categories_list = extract_categories(categories_answer)
                                categories = [point.strip() for point in categories_list if point.strip()]
                                categories = filter_categories(categories)

                            responses = get_responses_for_categories(categories, response_dict)
                            crime_category_response_list = "\n".join(f"- {responses[category]}" for category in categories if category in responses)
                            final_response_prompt = response_prompt_template + crime_category_response_list
                            response_prompt = PromptTemplate(template=final_response_prompt, input_variables=['context', 'question'])
                            response_chain = response_chain_define(response_prompt)
                            response = response_chain.invoke({"question": translated_prompt})
                            response_answer = response.get("answer", "")
                            # Replace all @tags with the username
                            response_answer = re.sub(r'@\w+', '', response_answer)
                        
                            response_note = extract_responses(response_answer, username)

                            # Priority Pipeline Starts Here
                            categories_str = ', '.join(categories)
                            print(categories_str)
                            final_priority_prompt = priority_prompt_template + 'The tweet belongs to the categories ' + categories_str + ' and has the following sentiment: ' + sentiment
                            priority_prompt = PromptTemplate(template=final_priority_prompt, input_variables=['context', 'question'])
                            priority_chain = priority_chain_define(priority_prompt)
                            priority = priority_chain.invoke({"question": translated_prompt})
                            priority_answer = priority.get("answer", "")

                            print(f"Priority answer: {priority_answer}")
                            priority_match = re.search(r'(High Priority|Low Priority)', priority_answer, re.IGNORECASE)
                            priority_assigned = priority_match.group(1).strip() if priority_match else "Priority Unavailable"
                            print(f"Priority assigned: {priority_assigned}")

                            if original_language in ['hi', 'hinglish']:
                                response_note = [translate_to_hindi(note['text']) for note in response_note]
                            else:
                                if isinstance(response_note, list):
                                    response_note = [note['text'] if isinstance(note, dict) else note for note in response_note]
                                else:
                                    response_note = [response_note]

                            draft = {
                                "id": tweet_data["tweet_id"],  # Using tweet_id as id for simplicity
                                "categories": categories,
                                "draft_reply": [response_note],  # Directly use the response note
                                "sentiment": sentiment,
                                "priority": priority_assigned
                            }
                            drafts_collection.insert_one(draft)
                            break  # Break the retry loop if successful
                    except Exception as e:
                        logger.error(f"Error processing tweet {tweet_data['tweet_id']}: {e}")
                        if "Already borrowed" in str(e):
                            retries -= 1
                            if retries > 0:
                                logger.info(f"Retrying tweet {tweet_data['tweet_id']}... {retries} retries left")
                                time.sleep(2)  # Delay before retrying
                            else:
                                logger.error(f"Failed to process tweet {tweet_data['tweet_id']} after retries")
                        else:
                            break  # Exit loop for non-retryable errors
                continue  # Move to the next tweet

        return jsonify({"message": "Tweets uploaded successfully."}), 200
    
    except Exception as e:
        logger.error(f"Error uploading tweets: {e}")
        return jsonify({"error": "An error occurred while uploading tweets"}), 500

@app.route('/test', methods=['GET'])
def test():
    try:
        return jsonify({"message": "Test route is working"})
    except Exception as e:
        logger.error(f"Error in test route: {e}")
        return jsonify({"error": "An error occurred in the test route"}), 500
    

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)