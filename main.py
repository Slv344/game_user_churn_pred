from  src.preprocessing_eda import  initial_exploration, define_features, plot_categorical, plot_numerical, preprocess_data
from src.model_training import train_models
import pandas as pd

def main():
    # Load and process data
    df = pd.read_csv('D://user_churn_prediction//Data//game_user_churn.csv')

    # Data exploration
    initial_exploration(df)
    numeric_features, categorical_features = define_features(df)

    # Visualize data
    plot_categorical(df)
    plot_numerical(df)

    # Preprocess data
    df_processed = preprocess_data(df)

    # Save processed data
    df_processed.to_csv('processed_game_user_churn.csv', index=False)

    # Train models
    train_models()

if __name__ == "__main__":
    main()
