import matplotlib.pyplot as plt
import ast
for i in range(1):
    with open(f"gate_grads_{i}.txt", 'r') as file:
        lines = file.readlines()

    values = [float(line.split("(")[1].split(")")[0]) for line in lines]

    if values:  
        epochs = list(range(1, len(values) + 1))

        # Plot the loss curve for the first 100 values
        plt.figure()
        plt.plot(epochs, values, marker='.', color='b', linewidth=1.5)
        plt.title(f'Gate_Gradients_{i} Gradients Values')
        plt.xlabel('Training Step')
        plt.ylabel('Gate Mean Gradient')
        # plt.ylim(1e-4, 1e-3)
        plt.savefig(f'gate_grad_{i}.png')
        plt.grid(True)
        # plt.legend()
        plt.show()
    else:
        print("No data to plot.")