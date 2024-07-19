# generates a heatmap with the scaled data
sns.heatmap(scaled_data.corr())

# run PCA to reduce the number of features
pca = PCA(n_components = 0.99)
pca.fit(scaled_data)
reduced = pca.transform(scaled_data)
reduced = pd.DataFrame(reduced)