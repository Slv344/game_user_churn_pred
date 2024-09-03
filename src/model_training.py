import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE

def print_scores(model_name, y_true, y_pred):
    print(f"{model_name} Model Performance:")
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("Precision: ", precision_score(y_true, y_pred))
    print("Recall: ", recall_score(y_true, y_pred))
    print("F1 Score: ", f1_score(y_true, y_pred))
    print("\n")

def train_models():
    # Load the data
    df = pd.read_csv('D://user_churn_prediction//processed_game_user_churn.csv')
    X = df.drop('churn', axis=1)
    y = df['churn']

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=50)

    # Logistic Regression
    log_model = LogisticRegression(max_iter=500, solver='liblinear', class_weight='balanced')
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    print_scores('Logistic Regression', y_test, y_pred_log)

    # SVM with Grid Search
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'class_weight': ['balanced']
    }
    svm_model = GridSearchCV(SVC(random_state=50), svm_params, cv=5, scoring='f1')
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    print_scores('SVM', y_test, y_pred_svm)

    # Random Forest with Grid Search
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }
    rf_model = GridSearchCV(RandomForestClassifier(random_state=50), rf_params, cv=5, scoring='f1')
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print_scores('Random Forest', y_test, y_pred_rf)

    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=50)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    print_scores('Decision Tree', y_test, y_pred_dt)

    # Evaluate which model performs best based on F1 Score
    models_performance = {
        'Logistic Regression': f1_score(y_test, y_pred_log),
        'SVM': f1_score(y_test, y_pred_svm),
        'Random Forest': f1_score(y_test, y_pred_rf),
        'Decision Tree': f1_score(y_test, y_pred_dt)
    }

    best_model_name = max(models_performance, key=models_performance.get)
    print(f"Best Model: {best_model_name} with F1 Score: {models_performance[best_model_name]}")

if __name__ == "__main__":
    train_models()