from flask import Flask, jsonify, request
from flask_cors import CORS
import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
import re
from collections import Counter, defaultdict
import json

# NLP and ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Network analysis
import networkx as nx

# Import OpenAI library
import openai
from openai import OpenAI

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# Load environment variables
load_dotenv()

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION')

MONGODB_ANALYSIS_DATABASE = os.getenv('MONGODB_ANALYSIS_DATABASE')
MONGODB_ANALYSIS_COLLECTION = os.getenv('MONGODB_ANALYSIS_COLLECTION')

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DATABASE]
collection = db[MONGODB_COLLECTION]

# New connection for analysis results
analysis_db = client[MONGODB_ANALYSIS_DATABASE]
analysis_collection = analysis_db[MONGODB_ANALYSIS_COLLECTION]

# Initialize NLP tools
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

# SME-related keywords for problem identification
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

class NewsAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.articles_df = None
        self.problem_clusters = {}
        
    def load_data(self):
        """Load data from MongoDB into pandas DataFrame"""
        cursor = collection.find({})
        articles = list(cursor)
        
        if not articles:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(articles)
        
        # Clean and process data
        df['content'] = df['content'].fillna('')
        df['title'] = df['title'].fillna('')
        df['combined_text'] = df['title'] + ' ' + df['content']
        
        # Convert word_count from MongoDB number format if needed
        if 'word_count' in df.columns:
            df['word_count'] = df['word_count'].apply(
                lambda x: int(x.get('$numberInt', 0)) if isinstance(x, dict) else x
            )
        
        # Parse publish_date
        df['publish_date_parsed'] = pd.to_datetime(df['publish_date'], errors='coerce')
        
        self.articles_df = df
        return df
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        dutch_stopwords = set(stopwords.words('dutch'))
        english_stopwords = set(stopwords.words('english'))
        stop_words = dutch_stopwords.union(english_stopwords)
        
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Lemmatize
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def identify_sme_problems(self, text):
        """Identify SME problems in text"""
        text_lower = text.lower()
        problems = []
        
        for problem_type, keywords in SME_PROBLEM_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    problems.append(problem_type)
                    break
        
        return list(set(problems))  # Remove duplicates
    
    def perform_sentiment_analysis(self, text):
        """Perform sentiment analysis on text"""
        if not isinstance(text, str) or not text.strip():
            return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
        
        return sia.polarity_scores(text)
    
    def cluster_articles(self, n_clusters=8):
        """Cluster articles using TF-IDF and K-means"""
        if self.articles_df is None or self.articles_df.empty:
            return {}
        
        # Preprocess text
        processed_texts = [self.preprocess_text(text) for text in self.articles_df['combined_text']]
        
        # Remove empty texts
        non_empty_indices = [i for i, text in enumerate(processed_texts) if text.strip()]
        processed_texts = [processed_texts[i] for i in non_empty_indices]
        
        if len(processed_texts) < n_clusters:
            n_clusters = max(1, len(processed_texts) // 2)
        
        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Create cluster mapping
        clusters = {}
        for i, label in enumerate(cluster_labels):
            original_idx = non_empty_indices[i]
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'article_id': str(self.articles_df.iloc[original_idx]['_id']),
                'title': self.articles_df.iloc[original_idx]['title'],
                'url': self.articles_df.iloc[original_idx]['url'],
                'domain': self.articles_df.iloc[original_idx]['domain']
            })
        
        # Get top terms for each cluster
        feature_names = self.vectorizer.get_feature_names_out()
        cluster_info = {}
        
        for i in range(n_clusters):
            if i in clusters:
                # Get top terms for this cluster
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-10:][::-1]
                top_terms = [feature_names[idx] for idx in top_indices]
                
                cluster_info[i] = {
                    'articles': clusters[i],
                    'top_terms': top_terms,
                    'size': len(clusters[i])
                }
        
        return cluster_info

analyzer = NewsAnalyzer()

@app.route('/')
def home():
    return jsonify({
        "message": "SME News Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/stats": "GET - Basic dataset statistics",
            "/articles": "GET - List all articles with filters",
            "/article/<id>": "GET - Get specific article details",
            "/sentiment": "GET - Sentiment analysis results",
            "/clusters": "GET - Article clustering results",
            "/problems": "GET - SME problems identification",
            "/network": "GET - Network analysis of problems and articles",
            "/search": "GET - Search articles by keyword"
        }
    })

@app.route('/stats')
def get_stats():
    """Get basic statistics about the dataset"""
    df = analyzer.load_data()
    
    if df.empty:
        return jsonify({"error": "No data found"}), 404
    
    # Convert word_count to numeric if it's in the new format
    if 'word_count' in df.columns:
        df['word_count'] = df['word_count'].apply(lambda x: int(x.get('$numberInt', 0)) if isinstance(x, dict) else x)
    
    stats = {
        "total_articles": len(df),
        "unique_domains": df['domain'].nunique(),
        "date_range": {
            "earliest": df['publish_date_parsed'].min().isoformat() if pd.notna(df['publish_date_parsed'].min()) else None,
            "latest": df['publish_date_parsed'].max().isoformat() if pd.notna(df['publish_date_parsed'].max()) else None
        },
        "top_domains": df['domain'].value_counts().head(10).to_dict(),
        "avg_word_count": float(df['word_count'].mean()) if 'word_count' in df.columns else None,
        "sme_related_stats": {
            "total_sme_related": int(df['is_sme_related'].sum()) if 'is_sme_related' in df.columns else None,
            "percentage_sme_related": float(df['is_sme_related'].mean() * 100) if 'is_sme_related' in df.columns else None
        }
    }
    
    return jsonify(stats)

@app.route('/articles')
def get_articles():
    """Get articles with optional filtering"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    domain = request.args.get('domain')
    is_sme_related = request.args.get('is_sme_related')
    
    query = {}
    if domain:
        query['domain'] = domain
    if is_sme_related is not None:
        query['is_sme_related'] = is_sme_related.lower() == 'true'
    
    # Get total count
    total = collection.count_documents(query)
    
    # Get paginated results
    articles = list(collection.find(query)
                   .skip((page - 1) * per_page)
                   .limit(per_page)
                   .sort('scraped_at', -1))
    
    # Convert ObjectId to string and ensure proper number types
    for article in articles:
        article['_id'] = str(article['_id'])
        if 'word_count' in article and isinstance(article['word_count'], dict):
            article['word_count'] = article['word_count'].get('$numberInt', 0)
    
    return jsonify({
        "articles": articles,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": (total + per_page - 1) // per_page
        }
    })

@app.route('/article/<article_id>')
def get_article(article_id):
    """Get specific article with analysis"""
    from bson import ObjectId
    
    try:
        article = collection.find_one({"_id": ObjectId(article_id)})
        if not article:
            return jsonify({"error": "Article not found"}), 404
        
        article['_id'] = str(article['_id'])
        if 'word_count' in article and isinstance(article['word_count'], dict):
            article['word_count'] = article['word_count'].get('$numberInt', 0)
        
        # Add existing sentiment analysis (VADER) and SME problem identification
        combined_text = (article.get('title', '') + ' ' + article.get('content', '')).strip()
        vader_sentiment = analyzer.perform_sentiment_analysis(combined_text)
        identified_problems = analyzer.identify_sme_problems(combined_text)

        article['analysis'] = {
            'vader_sentiment': vader_sentiment, # Keep VADER for comparison/option
            'identified_problems': identified_problems,
        }

        return jsonify(article)
        
    except Exception as e:
        spider.logger.error(f"Error processing article {article_id}: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/analysis/<article_id>')
def get_article_analysis(article_id):
    """Get advanced sentiment analysis for a specific article"""
    from bson import ObjectId

    try:
        # Find the analysis document in the new collection
        analysis_doc = analysis_collection.find_one({'article_id': article_id})

        if not analysis_doc:
            return jsonify({"error": "Analysis not found for this article"}), 404

        # Remove MongoDB's default _id field before returning
        if '_id' in analysis_doc:
            del analysis_doc['_id']

        return jsonify(analysis_doc)

    except Exception as e:
        print(f"Error retrieving analysis for article {article_id}: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/sentiment')
def get_sentiment_analysis():
    """Get sentiment analysis for all articles"""
    df = analyzer.load_data()
    
    if df.empty:
        return jsonify({"error": "No data found"}), 404
    
    sentiments = []
    for _, row in df.iterrows():
        combined_text = str(row.get('title', '')) + ' ' + str(row.get('content', ''))
        sentiment = analyzer.perform_sentiment_analysis(combined_text)
        
        sentiments.append({
            'article_id': str(row['_id']),
            'title': row.get('title', ''),
            'domain': row.get('domain', ''),
            'sentiment': sentiment,
            'is_sme_related': row.get('is_sme_related', False)
        })
    
    # Calculate overall statistics
    compounds = [s['sentiment']['compound'] for s in sentiments]
    sentiment_stats = {
        'overall_sentiment': {
            'mean_compound': float(np.mean(compounds)),
            'positive_articles': len([c for c in compounds if c > 0.05]),
            'negative_articles': len([c for c in compounds if c < -0.05]),
            'neutral_articles': len([c for c in compounds if -0.05 <= c <= 0.05])
        },
        'sme_related_sentiment': {
            'mean_compound': float(np.mean([s['sentiment']['compound'] for s in sentiments if s.get('is_sme_related', False)])),
            'positive_articles': len([s for s in sentiments if s.get('is_sme_related', False) and s['sentiment']['compound'] > 0.05]),
            'negative_articles': len([s for s in sentiments if s.get('is_sme_related', False) and s['sentiment']['compound'] < -0.05]),
            'neutral_articles': len([s for s in sentiments if s.get('is_sme_related', False) and -0.05 <= s['sentiment']['compound'] <= 0.05])
        },
        'articles': sentiments
    }
    
    return jsonify(sentiment_stats)

@app.route('/clusters')
def get_clusters():
    """Get article clustering results"""
    n_clusters = request.args.get('n_clusters', 8, type=int)
    
    df = analyzer.load_data()
    if df.empty:
        return jsonify({"error": "No data found"}), 404
    
    clusters = analyzer.cluster_articles(n_clusters)
    
    return jsonify({
        "n_clusters": len(clusters),
        "clusters": clusters
    })

@app.route('/problems')
def get_problems():
    """Get SME problems identification across all articles"""
    df = analyzer.load_data()
    
    if df.empty:
        return jsonify({"error": "No data found"}), 404
    
    problem_articles = defaultdict(list)
    problem_counts = Counter()
    sme_problem_counts = Counter()
    
    for _, row in df.iterrows():
        combined_text = str(row.get('title', '')) + ' ' + str(row.get('content', ''))
        problems = analyzer.identify_sme_problems(combined_text)
        is_sme_related = row.get('is_sme_related', False)
        
        for problem in problems:
            problem_counts[problem] += 1
            if is_sme_related:
                sme_problem_counts[problem] += 1
            problem_articles[problem].append({
                'article_id': str(row['_id']),
                'title': row.get('title', ''),
                'url': row.get('url', ''),
                'domain': row.get('domain', ''),
                'publish_date': row.get('publish_date', ''),
                'is_sme_related': is_sme_related
            })
    
    return jsonify({
        "problem_summary": {
            "all_articles": dict(problem_counts),
            "sme_related": dict(sme_problem_counts)
        },
        "problem_articles": dict(problem_articles),
        "total_articles_with_problems": len([row for _, row in df.iterrows() 
                                           if analyzer.identify_sme_problems(
                                               str(row.get('title', '')) + ' ' + str(row.get('content', ''))
                                           )]),
        "total_sme_articles_with_problems": len([row for _, row in df.iterrows() 
                                               if row.get('is_sme_related', False) and 
                                               analyzer.identify_sme_problems(
                                                   str(row.get('title', '')) + ' ' + str(row.get('content', ''))
                                               )])
    })

@app.route('/network')
def get_network_analysis():
    """Generate network analysis of problems and articles"""
    df = analyzer.load_data()
    
    if df.empty:
        return jsonify({"error": "No data found"}), 404
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes and edges
    problem_articles = defaultdict(list)
    article_problems = {}
    
    for _, row in df.iterrows():
        article_id = str(row['_id'])
        combined_text = str(row.get('title', '')) + ' ' + str(row.get('content', ''))
        problems = analyzer.identify_sme_problems(combined_text)
        is_sme_related = row.get('is_sme_related', False)
        
        if problems:
            # Add article node
            G.add_node(f"article_{article_id}", 
                      type='article', 
                      title=row.get('title', ''),
                      domain=row.get('domain', ''),
                      is_sme_related=is_sme_related)
            
            article_problems[article_id] = problems
            
            for problem in problems:
                # Add problem node
                G.add_node(f"problem_{problem}", type='problem', name=problem)
                
                # Add edge between article and problem
                G.add_edge(f"article_{article_id}", f"problem_{problem}",
                          connection_type='identifies_problem')
                
                problem_articles[problem].append(article_id)
    
    # Add edges between articles that share problems
    for problem, articles in problem_articles.items():
        if len(articles) > 1:
            for i, article1 in enumerate(articles):
                for article2 in articles[i+1:]:
                    if not G.has_edge(f"article_{article1}", f"article_{article2}"):
                        G.add_edge(f"article_{article1}", f"article_{article2}",
                                  connection_type='shared_problem',
                                  shared_problems=[problem])
                    else:
                        # Add to existing shared problems
                        edge_data = G.get_edge_data(f"article_{article1}", f"article_{article2}")
                        if 'shared_problems' in edge_data:
                            edge_data['shared_problems'].append(problem)
    
    # Calculate network metrics
    network_stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'connected_components': nx.number_connected_components(G),
        'sme_related_stats': {
            'num_sme_articles': len([n for n, d in G.nodes(data=True) 
                                   if d.get('type') == 'article' and d.get('is_sme_related', False)]),
            'num_sme_problem_connections': len([e for e in G.edges(data=True) 
                                              if e[2].get('connection_type') == 'identifies_problem' and
                                              G.nodes[e[0]].get('is_sme_related', False)])
        }
    }
    
    # Convert to serializable format
    nodes = []
    for node, data in G.nodes(data=True):
        node_info = {'id': node, **data}
        if data['type'] == 'problem':
            node_info['degree'] = G.degree(node)
        nodes.append(node_info)
    
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({'source': u, 'target': v, **data})
    
    return jsonify({
        'network_stats': network_stats,
        'nodes': nodes,
        'edges': edges
    })

@app.route('/network_analysis')
def get_advanced_network_analysis():
    """Generate advanced mathematical network analysis metrics"""
    df = analyzer.load_data()
    
    if df.empty:
        return jsonify({"error": "No data found"}), 404
    
    # Create network graph (same logic as /network)
    G = nx.Graph()
    
    problem_articles = defaultdict(list)
    
    for _, row in df.iterrows():
        article_id = str(row['_id'])
        combined_text = str(row.get('title', '')) + ' ' + str(row.get('content', ''))
        problems = analyzer.identify_sme_problems(combined_text)
        is_sme_related = row.get('is_sme_related', False)
        
        if problems:
            # Add article node
            G.add_node(f"article_{article_id}", 
                      type='article', 
                      title=row.get('title', ''),
                      domain=row.get('domain', ''),
                      is_sme_related=is_sme_related)
            
            for problem in problems:
                # Add problem node
                G.add_node(f"problem_{problem}", type='problem', name=problem)
                
                # Add edge between article and problem
                G.add_edge(f"article_{article_id}", f"problem_{problem}",
                          connection_type='identifies_problem')
                
                problem_articles[problem].append(article_id)
    
    # Add edges between articles that share problems
    for problem, articles in problem_articles.items():
        if len(articles) > 1:
            for i, article1 in enumerate(articles):
                for article2 in articles[i+1:]:
                    if not G.has_edge(f"article_{article1}", f"article_{article2}"):
                        G.add_edge(f"article_{article1}", f"article_{article2}",
                                  connection_type='shared_problem',
                                  shared_problems=[problem])
                    else:
                        # Add to existing shared problems
                        edge_data = G.get_edge_data(f"article_{article1}", f"article_{article2}")
                        if 'shared_problems' in edge_data:
                            edge_data['shared_problems'].append(problem)
                            
    # Perform advanced network analysis
    if G.number_of_nodes() == 0:
         return jsonify({"error": "Network has no nodes for analysis"}), 404
         
    try:
        avg_clustering = nx.average_clustering(G)
    except nx.NetworkXPointlessConcept:
        avg_clustering = 0.0
        
    try:
         degree_assortativity = nx.degree_assortativity_coefficient(G)
    except nx.NetworkXPointlessConcept:
         degree_assortativity = None # Or 0.0, depending on desired representation for no edges

    # Community Detection (using Girvan-Newman or similar, needs sorted edges or a faster algorithm)
    # For simplicity and performance on potentially larger graphs, using a greedy modularity maximization
    try:
        communities_generator = nx.community.greedy_modularity_communities(G)
        # Convert to list of lists of node IDs (strings)
        communities = [list(c) for c in communities_generator]
    except Exception as e:
        communities = f"Error calculating communities: {e}"

    # PageRank
    try:
        pagerank_scores = nx.pagerank(G)
        # Convert to list of tuples (node, score) for easier handling, sort by score
        sorted_pagerank = sorted(pagerank_scores.items(), key=lambda item: item[1], reverse=True)
    except Exception as e:
        pagerank_scores = f"Error calculating pagerank: {e}"
        sorted_pagerank = []

    # Structural Holes (Constraint and Effective Size)
    try:
        # Constraint measures the extent to which a node's contacts are connected to each other
        constraint_scores = nx.constraint(G)
        sorted_constraint = sorted(constraint_scores.items(), key=lambda item: item[1]) # Lower constraint is better for brokerage
    except Exception as e:
        constraint_scores = f"Error calculating constraint: {e}"
        sorted_constraint = []

    try:
        # Effective size measures the number of nonredundant contacts
        effective_size_scores = nx.effective_size(G)
        sorted_effective_size = sorted(effective_size_scores.items(), key=lambda item: item[1], reverse=True) # Higher effective size is better
    except Exception as e:
        effective_size_scores = f"Error calculating effective size: {e}"
        sorted_effective_size = []

    # Centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    
    # Summarize centrality measures by node type
    article_degree_centralities = [v for n, v in degree_centrality.items() if G.nodes[n].get('type') == 'article']
    problem_degree_centralities = [v for n, v in degree_centrality.items() if G.nodes[n].get('type') == 'problem']

    article_betweenness_centralities = [v for n, v in betweenness_centrality.items() if G.nodes[n].get('type') == 'article']
    problem_betweenness_centralities = [v for n, v in betweenness_centrality.items() if G.nodes[n].get('type') == 'problem']

    article_closeness_centralities = [v for n, v in closeness_centrality.items() if G.nodes[n].get('type') == 'article']
    problem_closeness_centralities = [v for n, v in closeness_centrality.items() if G.nodes[n].get('type') == 'problem']
    
    advanced_analysis_results = {
        'average_clustering_coefficient': avg_clustering,
        'degree_assortativity_coefficient': degree_assortativity,
        'centrality_measures_summary': {
            'article_nodes': {
                'degree_centrality': {
                    'mean': float(np.mean(article_degree_centralities)) if article_degree_centralities else 0.0,
                    'max': float(np.max(article_degree_centralities)) if article_degree_centralities else 0.0
                },
                 'betweenness_centrality': {
                    'mean': float(np.mean(article_betweenness_centralities)) if article_betweenness_centralities else 0.0,
                    'max': float(np.max(article_betweenness_centralities)) if article_betweenness_centralities else 0.0
                },
                'closeness_centrality': {
                    'mean': float(np.mean(article_closeness_centralities)) if article_closeness_centralities else 0.0,
                    'max': float(np.max(article_closeness_centralities)) if article_closeness_centralities else 0.0
                }
            },
            'problem_nodes': {
                 'degree_centrality': {
                    'mean': float(np.mean(problem_degree_centralities)) if problem_degree_centralities else 0.0,
                    'max': float(np.max(problem_degree_centralities)) if problem_degree_centralities else 0.0
                },
                 'betweenness_centrality': {
                    'mean': float(np.mean(problem_betweenness_centralities)) if problem_betweenness_centralities else 0.0,
                    'max': float(np.max(problem_betweenness_centralities)) if problem_betweenness_centralities else 0.0
                },
                'closeness_centrality': {
                    'mean': float(np.mean(problem_closeness_centralities)) if problem_closeness_centralities else 0.0,
                    'max': float(np.max(problem_closeness_centralities)) if problem_closeness_centralities else 0.0
                }
            }
        },
        'advanced_analysis': {
            'communities': communities,
            'pagerank_top_10': sorted_pagerank[:10],
            'constraint_top_10_lowest': sorted_constraint[:10],
            'effective_size_top_10': sorted_effective_size[:10]
        }
    }
    
    return jsonify(advanced_analysis_results)

@app.route('/search')
def search_articles():
    """Search articles by keyword"""
    query = request.args.get('q', '').strip()
    is_sme_related = request.args.get('is_sme_related')
    
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    
    # Build search query
    search_query = {
        "$or": [
            {"title": {"$regex": query, "$options": "i"}},
            {"content": {"$regex": query, "$options": "i"}}
        ]
    }
    
    if is_sme_related is not None:
        search_query['is_sme_related'] = is_sme_related.lower() == 'true'
    
    # MongoDB text search
    search_results = list(collection.find(search_query).limit(50))
    
    # Convert ObjectId to string and process word_count
    for result in search_results:
        result['_id'] = str(result['_id'])
        if 'word_count' in result and isinstance(result['word_count'], dict):
            result['word_count'] = result['word_count'].get('$numberInt', 0)
        
        # Add relevance score (simple keyword count)
        combined_text = (result.get('title', '') + ' ' + result.get('content', '')).lower()
        result['relevance_score'] = combined_text.count(query.lower())
    
    # Sort by relevance
    search_results.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return jsonify({
        "query": query,
        "total_results": len(search_results),
        "results": search_results
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)