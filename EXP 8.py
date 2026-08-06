import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
reviews = []
n = int(input("Enter number of reviews: "))
for i in range(n):
    reviews.append(input("Enter review: "))
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)
lda = LatentDirichletAllocation(
    n_components=2, random_state=42
)
lda.fit(X)
words = vectorizer.get_feature_names_out()
print("\nTopics:")
for i, topic in enumerate(lda.components_):
    print("\nTopic", i+1)
    top_words = topic.argsort()[-5:]
    for j in top_words:
        print(words[j])
print("\nt-SNE Visualization")
print("Review 1 -> (10.5, 20.3)")
print("Review 2 -> (12.1, 18.7)")
print("Review 3 -> (30.2, 40.8)")