import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import scrolledtext

class Cart:
    model = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=10, min_samples_leaf=5)
    scaler = MinMaxScaler()

    @classmethod
    def train(cls):
        df = pd.read_csv('data_4000v.csv')

        X = df[['qpa', 'pulse', 'respiratory_frequency']]  
        y = df['gravity_class']  
        
        X_normalized = cls.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

        cls.model.fit(X_train, y_train)

        y_pred = cls.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        cls.show_results(accuracy, precision, recall, f1, cm, classification_rep)

    @classmethod
    def predict_dataset(cls):
        new_df = pd.read_csv("data_800v.csv")

        X_new = new_df[['qpa', 'pulse', 'respiratory_frequency']]
        y_new = new_df['gravity_class']  

        X_new_normalized = cls.scaler.transform(X_new)

        predictions = cls.model.predict(X_new_normalized)

        accuracy = accuracy_score(y_new, predictions)
        precision = precision_score(y_new, predictions, average='weighted')
        recall = recall_score(y_new, predictions, average='weighted')
        f1 = f1_score(y_new, predictions, average='weighted')
        cm = confusion_matrix(y_new, predictions)

        classification_rep = classification_report(y_new, predictions)

        cls.show_results(accuracy, precision, recall, f1, cm, classification_rep)

    @staticmethod
    def show_results(accuracy, precision, recall, f1, cm, classification_rep):
        root = tk.Tk()
        root.title("Resultados do Modelo")

        text_area = scrolledtext.ScrolledText(root, width=100, height=30)
        text_area.pack(padx=10, pady=10)

        cm_str = '\n'.join(['\t'.join(map(str, row)) for row in cm])
        
        text_area.insert(tk.END, f'Accuracy: {accuracy:.2f}\n')
        text_area.insert(tk.END, f'Precision: {precision:.2f}\n')
        text_area.insert(tk.END, f'Recall: {recall:.2f}\n')
        text_area.insert(tk.END, f'F1-Score: {f1:.2f}\n')
        text_area.insert(tk.END, "Confusion Matrix:\n")
        text_area.insert(tk.END, cm_str + '\n')
        text_area.insert(tk.END, "\nClassification Report:\n")
        text_area.insert(tk.END, classification_rep)

        root.mainloop()

    @classmethod
    def predict(cls, file: str):
        new_data = pd.read_csv(file)
        X_new = new_data[['qpa', 'pulse', 'respiratory_frequency']]  
        X_new_scaled = cls.scaler.transform(X_new)
        y_pred = cls.model.predict(X_new_scaled)

        new_data['gravity_class'] = y_pred
        new_data.to_csv(file, index=False)

if __name__ == "__main__":
    Cart.train()
    Cart.predict_dataset()
