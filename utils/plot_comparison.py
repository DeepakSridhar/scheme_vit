import matplotlib.pyplot as plt

# Define the data for the plot
model_names_ = [
    ['Ours-s12', 'Ours-s24'],
    ['Pool-s12', 'Pool-s24', 'Pool-s36', 'Pool-m36', 'Pool-m48'],
    ['RSB-18', 'RSB-34', 'RSB-50', 'RSB-101', 'RSB-152'],
    ['DeiT-S', 'DeiT-B'],
    ['ResMLP-S12', 'ResMLP-S24', 'ResMLP-B24'],
    ['CoatNet-0', 'CoatNet-1', 'CoatNet-2', 'CoatNet-3']
    ]

model_names = [item for sublist in model_names_ for item in sublist]

accuracies_ = [
    [79.7, 81.9],
    [77.2, 80.3, 81.4, 82.1, 82.5],
    [70.6, 75.5, 79.8, 81.3, 81.8],
    [79.8, 81.8],
    [76.6, 79.4, 81.0],
    [81.6, 83.3, 84.1, 84.5],
]

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
nums = [2, 7, 12, 14, 17, 21]

accuracies = [item for sublist in accuracies_ for item in sublist]

flops_ = [
    [2.1, 4.1],
    [1.8, 3.4, 5.0, 8.8, 11.6],
    [1.8, 3.7, 4.1, 7.9, 11.6],
    [4.6, 17.5],
    [3.0, 6.0, 23.0],
    [4.2, 8.4, 15.7, 34.7],
]

flops = [item for sublist in flops_ for item in sublist]

params_ = [
    [11.83, 21.16],
    [12, 21, 31, 56, 73],
    [12, 22, 26, 45, 60],
    [22, 86],
    [15, 30, 116],
    [25, 42, 75, 168],
]

params = [item for sublist in params_ for item in sublist]

# # Create a scatter plot of accuracy vs FLOPS
for i, name in enumerate(model_names_):
    plt.scatter(flops_[i], accuracies_[i], color=colors[i])
    plt.plot(flops_[i], accuracies_[i], '--', color=colors[i])

# Add axis labels and a title
plt.xlabel('FLOPS (G)')
plt.ylabel('Imagenet Top-1 Val Accuracy (%)')
plt.title('Accuracy vs FLOPS')

# Add a label for each data point
k = 0
arrowstyle = {"arrowstyle" : "-", "linestyle" : "--",
            "shrinkA": 0, "shrinkB": 0}
for i, name in enumerate(model_names):
    if i < nums[k]:
        pass
    else:
        k = k + 1
    color = colors[k]
    
    plt.annotate(name, (flops[i], accuracies[i]), color=color)

# Show the plot
plt.savefig('AccuracyFLOPs.png')

plt.figure()
# # Create a scatter plot of accuracy vs Params
for i, name in enumerate(model_names_):
    plt.scatter(params_[i], accuracies_[i], color=colors[i])
    plt.plot(params_[i], accuracies_[i], '--', color=colors[i])

# Add axis labels and a title
plt.xlabel('Params (M)')
plt.ylabel('Imagenet Top-1 Val Accuracy (%)')
plt.title('Accuracy vs Params')

# Add a label for each data point
k = 0
arrowstyle = {"arrowstyle" : "-", "linestyle" : "--",
            "shrinkA": 0, "shrinkB": 0}
for i, name in enumerate(model_names):
    if i < nums[k]:
        pass
    else:
        k = k + 1
    color = colors[k]
    
    plt.annotate(name, (params[i], accuracies[i]), color=color)

# Show the plot
plt.savefig('AccuracyParams.png')
