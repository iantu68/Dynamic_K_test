import matplotlib.pyplot as plt

# Data
data = {'0': 5864742,
'1': 10806993,
'2': 8129672,
'3': 23899507,
'4': 42307600,
'5': 5183870,
'6': 24886997,
'7': 14893486,
}
# Create bar chart
plt.figure(figsize=(10,6))
plt.bar(range(len(data)), data.values(), align='center')
plt.xticks(range(len(data)), list(data.keys()))
plt.xlabel('Gate Number')
plt.ylabel('Count')
plt.title('Bert_Tiny_Layer1')
plt.ylim(0, 50000000)
# Save plot as a PDF file
plt.savefig('Bert_Tiny_Layer1', bbox_inches='tight')
plt.show()
