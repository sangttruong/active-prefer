import matplotlib.pyplot as plt
import numpy as np
from statistics import pvariance
import seaborn as sns
import pandas as pd

def plot_acc():
    data = [
        {'label': 'Random', 'data': [{'iter': 0, 'acc': 0.6341463414634146}, {'iter': 1, 'acc': 0.6029268292682927}, {'iter': 2, 'acc': 0.5892682926829268}, {'iter': 3, 'acc': 0.577560975609756}, {'iter': 4, 'acc': 0.5912195121951219}]},
        {'label': 'Max_entropy', 'data': [{'iter': 0, 'acc': 0.6302439024390244}, {'iter': 1, 'acc': 0.6136585365853658}, {'iter': 2, 'acc': 0.5941463414634146}, {'iter': 3, 'acc': 0.577560975609756}, {'iter': 4, 'acc': 0.6009756097560975}]}
    ]

    plt.figure()
    for dataset in data:
        x = [point['iter'] for point in dataset['data']]
        y = [point['acc'] for point in dataset['data']]
        plt.plot(x, y, label=dataset['label'])

    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iteration')
    plt.legend()
    plt.savefig("images/llama_reward_bench.png")
    plt.show()

def plot_oracle_acc():
    metrics = [
        {"model_id": i, "Accuracy": 0.82 + i * 0.001} for i in range(10)
    ]
    accuracies = [metric["Accuracy"] for metric in metrics]
    mean_accuracy = np.mean(accuracies)
    variance_accuracy = pvariance(accuracies)

    plt.figure(figsize=(8, 6))
    plt.hist(accuracies, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=mean_accuracy, color='red', linestyle='--', label=f'Mean: {mean_accuracy:.2f}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Model Accuracies')
    plt.legend()
    plt.grid(True)
    plt.annotate(f'Mean Accuracy: {mean_accuracy:.2f}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=12)
    plt.annotate(f'Variance of Accuracy: {variance_accuracy:.1e}', xy=(0.5, 0.85), xycoords='axes fraction', ha='center', fontsize=12)
    plt.savefig('images/accuracy_histogram.png')
    plt.show()

def plot_model_performance(model_dict):
    models = list(model_dict.keys())
    scores = list(model_dict.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color='skyblue', width=0.5)
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Performance')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.6, 1)

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{score:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"images/oracle_acc.png")
    plt.show()

def plot_bar_chart(data):
    df = pd.DataFrame(data, columns=['Dataset', 'Model', 'Mean Accuracy', 'Variance'])
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Model', y='Mean Accuracy', color='skyblue', label='Mean Accuracy')
    sns.barplot(data=df, x='Model', y='Variance', color='lightgreen', label='Variance')

    plt.xlabel('Model')
    plt.ylabel('Scores')
    plt.title(f'Mean Accuracy and Variance for Different Models on {data[0][0]}')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left')
    plt.ylim(0.5, 1.1)

    autolabel(plt.gca().patches, df['Mean Accuracy'], df['Variance'])

    dataset_name = data[0][0].replace('/', '-')
    filename = f'images/{dataset_name}.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def main():
    results = [
        ("ARC-Challenge", "mistralai/Mistral-7B-v0.1", 0.71, 6.30E-04),
        ("ARC-Challenge", "mistralai/Mistral-7B-Instruct-v0.2", 0.72, 4.06E-04),
        # Add more results as needed
    ]

    for res in results:
        plot_bar_chart(res)

if __name__ == "__main__":
    main()

