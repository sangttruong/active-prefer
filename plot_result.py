import matplotlib.pyplot as plt
import numpy as np

def plot_acc():
    # Data
    random_data = [{'iter': 0, 'acc': 0.6341463414634146},
                {'iter': 1, 'acc': 0.6029268292682927},
                {'iter': 2, 'acc': 0.5892682926829268},
                {'iter': 3, 'acc': 0.577560975609756},
                {'iter': 4, 'acc': 0.5912195121951219}]

    max_entropy_data = [{'iter': 0, 'acc': 0.6302439024390244},
                        {'iter': 1, 'acc': 0.6136585365853658},
                        {'iter': 2, 'acc': 0.5941463414634146},
                        {'iter': 3, 'acc': 0.577560975609756},
                        {'iter': 4, 'acc': 0.6009756097560975}]

    # Extracting x and y values
    random_x = [point['iter'] for point in random_data]
    random_y = [point['acc'] for point in random_data]

    max_entropy_x = [point['iter'] for point in max_entropy_data]
    max_entropy_y = [point['acc'] for point in max_entropy_data]

    # Plotting
    plt.plot(random_x, random_y, label='Random')
    plt.plot(max_entropy_x, max_entropy_y, label='Max_entropy')

    # Adding labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iteration')
    plt.legend()

    # Displaying the plot
    plt.show()
    plt.savefig("images/llama_reward_bench.png")


def plot_oracle_acc():
    # metrics = [{'model_id': 0, 'loss': 0.6969875154788034, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_0.safetensors'}, {'model_id': 1, 'loss': 0.5134088109459793, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_1.safetensors'}, {'model_id': 2, 'loss': 0.4989338775998668, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_2.safetensors'}, {'model_id': 3, 'loss': 0.4770105308608005, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_3.safetensors'}, {'model_id': 4, 'loss': 0.5097642797127104, 'Accuracy': 0.8571428571428571, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_4.safetensors'}, {'model_id': 5, 'loss': 0.5076179802417755, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_5.safetensors'}, {'model_id': 6, 'loss': 0.4847733150971563, 'Accuracy': 0.7142857142857143, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_6.safetensors'}, {'model_id': 7, 'loss': 0.5075322198763228, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_7.safetensors'}, {'model_id': 8, 'loss': 0.518027106920878, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_8.safetensors'}, {'model_id': 9, 'loss': 0.5117164905134001, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_9.safetensors'}, {'model_id': 10, 'loss': 0.4988821410296256, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_10.safetensors'}, {'model_id': 11, 'loss': 0.5087874319992567, 'Accuracy': 0.5714285714285714, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_11.safetensors'}, {'model_id': 12, 'loss': 0.504003120618954, 'Accuracy': 0.8571428571428571, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_12.safetensors'}, {'model_id': 13, 'loss': 0.4922906165583092, 'Accuracy': 0.8571428571428571, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_13.safetensors'}, {'model_id': 14, 'loss': 0.5006412223242876, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_14.safetensors'}, {'model_id': 15, 'loss': 0.49743010391268816, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_15.safetensors'}, {'model_id': 16, 'loss': 0.49835661811786786, 'Accuracy': 0.5714285714285714, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_16.safetensors'}, {'model_id': 17, 'loss': 0.5118268775312524, 'Accuracy': 0.8571428571428571, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_17.safetensors'}, {'model_id': 18, 'loss': 0.48823955576670797, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_18.safetensors'}, {'model_id': 19, 'loss': 0.4786361414089538, 'Accuracy': 0.42857142857142855, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_19.safetensors'}, {'model_id': 20, 'loss': 0.48317905450076387, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_20.safetensors'}, {'model_id': 21, 'loss': 0.47974638614738196, 'Accuracy': 0.8571428571428571, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_21.safetensors'}, {'model_id': 22, 'loss': 0.4902948595975575, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_22.safetensors'}, {'model_id': 23, 'loss': 0.5014702499958507, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_23.safetensors'}, {'model_id': 24, 'loss': 0.5077021919321596, 'Accuracy': 0.8571428571428571, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_24.safetensors'}, {'model_id': 25, 'loss': 0.518485146656371, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_25.safetensors'}, {'model_id': 26, 'loss': 0.516034486000998, 'Accuracy': 0.7142857142857143, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_26.safetensors'}, {'model_id': 27, 'loss': 0.48904681807024436, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_27.safetensors'}, {'model_id': 28, 'loss': 0.5146132874907109, 'Accuracy': 1.0, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_28.safetensors'}, {'model_id': 29, 'loss': 0.48006389747586165, 'Accuracy': 0.8571428571428571, 'model_path': 'saves/Llama-2-7b-hf/reward_bench_train_Llama-2-7b-hf_max_entropy_check/max_entropy/oracle_29.safetensors'}]
    # Extract accuracy values from the metrics
    # accuracies = [metric["Accuracy"] for metric in metrics]
    
    accuracies = [0.5971, ]
    
    # Calculate mean and variance of accuracy
    mean_accuracy = np.mean(accuracies)
    variance_accuracy = np.var(accuracies)

    # Plot the image
    plt.figure(figsize=(8, 6))
    plt.hist(accuracies, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=mean_accuracy, color='red', linestyle='--', label=f'Mean: {mean_accuracy:.2f}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Model Accuracies')
    plt.legend()
    plt.grid(True)

    # Annotate mean and variance on the plot
    plt.annotate(f'Mean Accuracy: {mean_accuracy:.2f}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=12)
    plt.annotate(f'Variance of Accuracy: {variance_accuracy:.2f}', xy=(0.5, 0.85), xycoords='axes fraction', ha='center', fontsize=12)


    plt.savefig('images/accuracy_histogram.png')
    plt.show()

    print(f"Mean Accuracy: {mean_accuracy:.2f}")
    print(f"Variance of Accuracy: {variance_accuracy:.2f}")


plot_oracle_acc()