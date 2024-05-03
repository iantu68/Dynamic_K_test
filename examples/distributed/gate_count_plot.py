import matplotlib.pyplot as plt

# Data
data = {'0': 18694644,
'1': 3495007,
'2': 10116996,
'3': 2916110,
'4': 5940428,
'5': 18747012,
'6': 1970231,
'7': 6106004}
# Create bar chart
plt.figure(figsize=(10,6))
plt.bar(range(len(data)), data.values(), align='center')
plt.xticks(range(len(data)), list(data.keys()))
plt.xlabel('Gate Number')
plt.ylabel('Count')
plt.title('Bert_Tiny_Layer0')
plt.ylim(0, 15000000)
# Save plot as a PDF file
plt.savefig('Bert_Tiny_Layer0', bbox_inches='tight')
plt.show()
