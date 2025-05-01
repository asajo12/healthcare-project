# 1. Heart Disease Frequency
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='target', data=df, palette='Set2')
legend_labels = ['No Heart Disease', 'Has Heart Disease']
handles = [plt.Rectangle((0,0),1,1, color=c) for c in sns.color_palette('Set2')[:2]]
plt.legend(handles, legend_labels, title="Condition")
plt.title('Heart Disease Frequency')
plt.xlabel('Target')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2. Age distribution by target
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='age', hue='target', multiple='stack', palette='Set1', bins=20)
plt.title('Age Distribution by Heart Disease Status')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 3. Cholesterol vs. Age, colored by target
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='age', y='chol', hue='target', palette='coolwarm')
plt.title('Cholesterol vs Age by Heart Disease Status')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.tight_layout()
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


