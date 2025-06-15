import matplotlib.pyplot as plt

# Hardcoded data for 9 models
model_names = [
    "ViT", "Swin", "DeiT", 
    "BEiT", "CvT", "Resnet-18", 
    "Resnet-50", "Resnet-152", "ConvNeXt"
]

throughputs = [52.23, 17.67, 51.71, 37.39, 38.90, 159.25, 87.13, 34.77, 55.72]  # example values
accuracies = [97.06, 98.01, 97.20, 97.77, 97.57, 96.31, 96.41, 97.54, 97.95]  # example values

# Bar plot: Throughput per model
plt.figure(figsize=(10, 6))
plt.bar(model_names, throughputs)
plt.title("Throughput per Model")
plt.xlabel("Model")
plt.ylabel("Throughput (images/s)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('throughput.pdf', bbox_inches='tight')
plt.show()

# Scatter plot: Accuracy vs Throughput
plt.figure(figsize=(8, 6))
plt.scatter(throughputs, accuracies)

# Annotate each point with model name
for i, name in enumerate(model_names):
    plt.annotate(name, (throughputs[i], accuracies[i]))

plt.title("Accuracy vs Throughput")
plt.xlabel("Throughput (images/s)")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_vs_throughput.pdf', bbox_inches='tight')
plt.show()
