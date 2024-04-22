import matplotlib.pyplot as plt

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
