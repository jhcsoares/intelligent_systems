import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class NeuralNetwork:
    scaler = StandardScaler()
    model = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=5000)

    @classmethod
    def train(cls):
        df = pd.read_csv('4000v.csv')

        X = df[['qpa', 'pulse', 'respiratory_frequency']]  
        y = df['gravity_class']                      

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train = cls.scaler.fit_transform(X_train)
        X_test = cls.scaler.transform(X_test)

        cls.model.fit(X_train, y_train)

        y_pred = cls.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

    @classmethod
    def test(cls):
        new_data = pd.read_csv('225_env_vital_signals.csv')
        X_new = new_data[['qpa', 'pulse', 'respiratory_frequency']]  
        y_true = new_data['gravity_class'] 
        X_new_scaled = cls.scaler.transform(X_new)
        y_pred = cls.model.predict(X_new_scaled)

        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
