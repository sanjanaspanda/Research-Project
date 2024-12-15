import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class LonelinessNLPAnalysis:
    def __init__(self, data):
        self.df = data.copy()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Comprehensive text cleaning method
        """
        if pd.isna(text):
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(cleaned_tokens)
    
    def perform_sentiment_analysis(self, text_column):
        """
        Conduct sentiment analysis on a text column
        """
        # Clean the text first
        self.df[f'{text_column}_cleaned'] = self.df[text_column].apply(self.clean_text)
        
        # Perform sentiment analysis
        def get_sentiment(text):
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'sentiment_category': self._categorize_sentiment(blob.sentiment.polarity)
            }
        
        sentiment_results = self.df[f'{text_column}_cleaned'].apply(get_sentiment)
        
        # Expand sentiment results into separate columns
        self.df[f'{text_column}_polarity'] = sentiment_results.apply(lambda x: x['polarity'])
        self.df[f'{text_column}_subjectivity'] = sentiment_results.apply(lambda x: x['subjectivity'])
        self.df[f'{text_column}_sentiment_category'] = sentiment_results.apply(lambda x: x['sentiment_category'])
    
    def _categorize_sentiment(self, polarity):
        """
        Categorize sentiment polarity
        """
        if polarity > 0.05:
            return 'Positive'
        elif polarity < -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    def extract_key_themes(self, text_column, n_themes=5):
        """
        Extract key themes using TF-IDF and K-means clustering
        """
        # Clean the text
        cleaned_texts = self.df[text_column].apply(self.clean_text)
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        
        # K-means clustering to identify themes
        kmeans = KMeans(n_clusters=n_themes, random_state=42)
        kmeans.fit(tfidf_matrix)
        
        # Get top words for each theme
        feature_names = vectorizer.get_feature_names_out()
        theme_keywords = {}
        
        for i in range(n_themes):
            # Get the indices of the closest documents to the cluster center
            closest_docs = np.argsort(np.linalg.norm(tfidf_matrix - kmeans.cluster_centers_[i], axis=1))[:5]
            
            # Get top words for this theme
            theme_words = []
            for doc_idx in closest_docs:
                doc_vector = tfidf_matrix[doc_idx].toarray()[0]
                top_word_indices = doc_vector.argsort()[-5:][::-1]
                theme_words.extend([feature_names[idx] for idx in top_word_indices])
            
            theme_keywords[f'Theme {i+1}'] = list(set(theme_words))
        
        return theme_keywords
    
    def visualize_sentiment_distribution(self, text_column):
        """
        Visualize sentiment distribution across different age groups
        """
        # Ensure sentiment analysis is performed
        if f'{text_column}_sentiment_category' not in self.df.columns:
            self.perform_sentiment_analysis(text_column)
        sentiment_by_age = pd.crosstab(
            self.df['Age'], 
            self.df[f'{text_column}_sentiment_category'], 
            normalize='index'
        )
        
        # Plotting
        plt.figure(figsize=(10, 6))
        sentiment_by_age.plot(kind='bar', stacked=True)
        plt.title(f'Sentiment Distribution for {text_column} Across Age Groups')
        plt.xlabel('Age Group')
        plt.ylabel('Proportion')
        plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('sentiment_distribution.png')
        
        return sentiment_by_age
    
    def comprehensive_text_analysis(self):
        """
        Perform comprehensive NLP analysis on open-ended questions
        """
        # Open-ended columns from the original research
        open_ended_cols = [
            'How has technology influenced your sense of community and personal connections?',
            'Describe a meaningful online or offline community experience that impacted your wellbeing.'
        ]
        
        # Results dictionary to store all analyses
        nlp_analysis_results = {}
        
        for column in open_ended_cols:
            # Sentiment Analysis
            self.perform_sentiment_analysis(column)
            
            # Theme Extraction
            themes = self.extract_key_themes(column)
            
            # Sentiment Distribution
            sentiment_dist = self.visualize_sentiment_distribution(column)
            
            nlp_analysis_results[column] = {
                'sentiment_distribution': sentiment_dist,
                'key_themes': themes,
                'overall_sentiment': {
                    'positive_percentage': (self.df[f'{column}_sentiment_category'] == 'Positive').mean(),
                    'negative_percentage': (self.df[f'{column}_sentiment_category'] == 'Negative').mean(),
                    'neutral_percentage': (self.df[f'{column}_sentiment_category'] == 'Neutral').mean()
                }
            }
        
        return nlp_analysis_results

# Example usage
if __name__ == '__main__':
    # Load your data
    data = pd.read_csv('Loneliness, Technology, and Community Wellbeing Survey (Responses) - Form Responses 1.csv')
    data.fillna(method='ffill', inplace=True)
    data.columns = data.columns.str.strip()
    
    # Create NLP analysis instance
    nlp_analysis = LonelinessNLPAnalysis(data)
    
    # Perform comprehensive text analysis
    results = nlp_analysis.comprehensive_text_analysis()
    
    # Print results
    for column, analysis in results.items():
        print(f"\nAnalysis for: {column}")
        print("Sentiment Distribution:")
        print(analysis['sentiment_distribution'])
        print("\nKey Themes:")
        for theme, keywords in analysis['key_themes'].items():
            print(f"{theme}: {keywords}")
        print("\nOverall Sentiment:")
        print(analysis['overall_sentiment'])