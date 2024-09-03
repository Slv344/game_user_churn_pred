import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv('D://user_churn_prediction//Data//game_user_churn.csv')

# Initial data exploration
def initial_exploration(df):
    print(df.head(5))
    print(df.shape)
    print(df.isnull().sum())
    print(df.nunique())
    print(df.describe())
    print(df.info())

    # Print unique categories for categorical features
    for col in ['gender', 'country', 'game_genre', 'subscription_status', 'device_type', 'favorite_game_mode']:
        print(f"Categories in '{col}' variable: {df[col].unique()}")

# Define numerical and categorical columns
def define_features(df):
    numeric_features = [feature for feature in df.columns if df[feature].dtype in ['int64', 'float64']]
    categorical_features = [feature for feature in df.columns if df[feature].dtype == 'object']

    print('We have {} numerical features: {}'.format(len(numeric_features), numeric_features))
    print('\nWe have {} categorical features: {}'.format(len(categorical_features), categorical_features))

    return numeric_features, categorical_features

# Plot categorical features
def plot_categorical(df):
    # Plotting Churn Distribution
    target_instance = df['churn'].value_counts().reset_index()
    target_instance.columns = ['churn', 'count']
    
    fig = px.pie(target_instance, values='count', names='churn', 
                 color_discrete_sequence=["green", "red"],
                 title='Distribution of Churn')
    fig.show()

    # Plot categorical variables against churn
    categorical_cols = ['gender', 'subscription_status', 'device_type', 'country', 'favorite_game_mode', 'game_genre']
    for col in categorical_cols:
        sns.barplot(x=col, y='churn', data=df)
        plt.show()

# Plot numerical features
def plot_numerical(df):
    # Boxplot for total_play_time vs Churn
    sns.boxplot(x='churn', y='total_play_time', data=df)
    plt.show()

    # Histogram for avg_session_time vs Churn
    sns.histplot(df[df['churn'] == 0]['avg_session_time'], color='blue', label='Not Churned', kde=True)
    sns.histplot(df[df['churn'] == 1]['avg_session_time'], color='red', label='Churned', kde=True)
    plt.legend()
    plt.show()

   

# Data preprocessing
def preprocess_data(df):
    df['subscription_status'] = df['subscription_status'].map({'Yes': 1, 'No': 0})
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})
    
    # One-hot encoding categorical variables
    df = pd.get_dummies(df, columns=['game_genre', 'country', 'device_type', 'favorite_game_mode'], drop_first=True)
    
    # Scaling numerical features
    numerical_features = [
        'total_play_time', 'avg_session_time', 'games_played', 
        'in_game_purchases', 'last_login', 'friend_count', 
        'max_level_achieved', 'daily_play_time', 'number_of_sessions', 
        'social_interactions', 'achievement_points'
    ]
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df

# Execute the functions
initial_exploration(df)
numeric_features, categorical_features = define_features(df)
plot_categorical(df)
plot_numerical(df)
df = preprocess_data(df)

# Save the processed data for later use
df.to_csv('processed_game_user_churn.csv', index=False)
