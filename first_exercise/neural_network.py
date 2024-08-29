import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import tkinter as tk
from tkinter import scrolledtext

class NeuralNetwork:
    scaler = MinMaxScaler()
    model = MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation="relu", solver="adam", max_iter=30000, momentum=0.9)

    @classmethod
    def train(cls):
        df = pd.read_csv('data_4000v.csv')

        X = df[['qpa', 'pulse', 'respiratory_frequency']]  
        y = df['gravity_class']                      

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train = cls.scaler.fit_transform(X_train)
        X_test = cls.scaler.transform(X_test)

        cls.model.fit(X_train, y_train)

        # Fazer previsões no conjunto de teste
        y_pred = cls.model.predict(X_test)

        # Calcular as métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        # Exibir os resultados em uma janela tkinter
        cls.show_results(accuracy, precision, recall, f1, cm, classification_rep)

    @classmethod
    def predict(cls, file: str):
        new_data = pd.read_csv(file)
        X_new = new_data[['qpa', 'pulse', 'respiratory_frequency']]  
        X_new_scaled = cls.scaler.transform(X_new)
        y_pred = cls.model.predict(X_new_scaled)

        new_data['gravity_class'] = y_pred
        new_data.to_csv(file, index=False)

    @classmethod
    def predict_dataset(cls, file: str):
        # Carregar e processar os novos dados
        new_df = pd.read_csv(file)
        X_new = new_df[['qpa', 'pulse', 'respiratory_frequency']]  
        y_new = new_df['gravity_class']  

        X_new_scaled = cls.scaler.transform(X_new)

        # Fazer as previsões
        predictions = cls.model.predict(X_new_scaled)

        # Calcular as métricas
        accuracy = accuracy_score(y_new, predictions)
        precision = precision_score(y_new, predictions, average='weighted')
        recall = recall_score(y_new, predictions, average='weighted')
        f1 = f1_score(y_new, predictions, average='weighted')
        cm = confusion_matrix(y_new, predictions)
        classification_rep = classification_report(y_new, predictions)

        # Exibir os resultados em uma janela tkinter
        cls.show_results(accuracy, precision, recall, f1, cm, classification_rep)

    @classmethod
    def show_results(cls, accuracy, precision, recall, f1, cm, classification_rep):
        # Criar uma janela do tkinter
        root = tk.Tk()
        root.title("Resultados da Rede Neural")

        # Adicionar um widget de texto com rolagem
        text_area = scrolledtext.ScrolledText(root, width=100, height=30)
        text_area.pack(padx=10, pady=10)

        # Formatando a matriz de confusão
        cm_str = '\n'.join(['\t'.join(map(str, row)) for row in cm])

        # Adicionar os resultados ao widget de texto
        text_area.insert(tk.END, f'Accuracy: {accuracy:.2f}\n')
        text_area.insert(tk.END, f'Precision: {precision:.2f}\n')
        text_area.insert(tk.END, f'Recall: {recall:.2f}\n')
        text_area.insert(tk.END, f'F1-Score: {f1:.2f}\n')
        text_area.insert(tk.END, "Confusion Matrix:\n")
        text_area.insert(tk.END, cm_str + '\n')
        text_area.insert(tk.END, "\nClassification Report:\n")
        text_area.insert(tk.END, classification_rep)

        # Iniciar o loop principal do tkinter
        root.mainloop()

if __name__ == "__main__":
    NeuralNetwork.train()
    NeuralNetwork.predict_dataset('data_800v.csv')
