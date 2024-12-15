import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

class LonelinessInsightAnalysis:
    def __init__(self, data):
        self.df = data.copy()
        
    def prepare_data(self):
        # Preprocessing similar to original code
        feature_columns = [
            'Average daily screen time (hours)',
            'Communicate with friends',
            'Communicate with family',
            'Meet new people',
            'Professional networking',
            'Online communities/forums',
            'How frequently do you feel lonely?',
            'I feel deeply connected to my local community',
            'Online interactions satisfy my social needs',
            'I have meaningful relationships',
            'Technology helps me maintain relationships',
            'I feel isolated despite being digitally connected'
        ]
        
        # Encode categorical variables
        le = LabelEncoder()
        for col in feature_columns:
            if self.df[col].dtype == 'object':
                self.df[col] = le.fit_transform(self.df[col].astype(str))
        
        # Prepare target variable
        def categorize_wellbeing(value):
            if value in ['No Impact', 'Slight Impact']:
                return 'Low'
            elif value in ['Moderate Impact']:
                return 'Moderate'
            else:
                return 'High'
        
        self.df['Wellbeing'] = self.df['Sense of belonging'].apply(categorize_wellbeing)
        
        # Select features and target
        X = self.df[feature_columns]
        y = le.fit_transform(self.df['Wellbeing'])
        
        return X, y
    
    def analyze_feature_importance(self):
        # Prepare data
        X, y = self.prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Compute feature importances
        importances = rf.feature_importances_
        feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        
        # Visualize feature importances
        plt.figure(figsize=(10, 6))
        feature_importances.plot(kind='bar')
        plt.title('Feature Importances in Wellbeing Prediction')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        # Permutation importance for robustness
        perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
        perm_feature_importances = pd.Series(perm_importance.importances_mean, index=X.columns).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        perm_feature_importances.plot(kind='bar')
        plt.title('Permutation Feature Importances')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        plt.savefig('permutation_importance.png')
        
        return feature_importances, perm_feature_importances
    
    def age_group_loneliness_analysis(self):
        # Analyze loneliness by age group
        loneliness_by_age = self.df.groupby('Age')['How frequently do you feel lonely?'].value_counts(normalize=True).unstack()
        loneliness_by_age.fillna('0', inplace=True)
        plt.figure(figsize=(10, 6))
        loneliness_by_age.plot(kind='bar', stacked=True)
        plt.title('Loneliness Frequency by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Proportion')
        plt.legend(title='Loneliness Frequency', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('loneliness_by_age.png')
        
        return loneliness_by_age

# Example usage
if __name__ == '__main__':
    # Load your data
    data = pd.read_csv('Loneliness, Technology, and Community Wellbeing Survey (Responses) - Form Responses 1.csv')
    data.fillna(method='ffill', inplace=True)
    data.columns = data.columns.str.strip()
    
    # Create analysis instance
    analysis = LonelinessInsightAnalysis(data)
    
    # Analyze feature importance
    feature_importances, perm_importances = analysis.analyze_feature_importance()
    print("Tree-based Feature Importances:")
    print(feature_importances)
    print("\nPermutation Feature Importances:")
    print(perm_importances)
    
    # Analyze loneliness by age group
    loneliness_by_age = analysis.age_group_loneliness_analysis()
    print("\nLoneliness Frequency by Age Group:")
    print(loneliness_by_age)