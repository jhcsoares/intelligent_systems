import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class Cart:
    cart_model = DecisionTreeClassifier(random_state=42)

    @classmethod
    def train(cls):
        df = pd.read_csv('4000v.csv')

        X = df[['qpa', 'pulse', 'respiratory_frequency']]  
        y = df['gravity_class']  

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        cls.cart_model.fit(X_train, y_train)

        y_pred = cls.cart_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

