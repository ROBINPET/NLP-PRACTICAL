import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

try:
    wn.synsets("computer")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

print("===== NLP Lab Experiment 3 =====")
print("1. Text Similarity using TF-IDF + Cosine Similarity")
print("2. Clustering Headlines")
print("3. WordNet Similarity")
print("================================")

headlines = []
n = int(input("Enter the number of headlines: "))
for i in range(n):
    headline = input(f"Enter headline {i + 1}: ")
    headlines.append(headline)

vectorizer = TfidfVectorizer(stop_words="english")
x = vectorizer.fit_transform(headlines)

print("\nCosine Similarity Matrix:")
print(cosine_similarity(x))

kmeans = KMeans(n_clusters=min(2, len(headlines)), random_state=0, n_init=10)
kmeans.fit(x)

print("\nHeadline Clusters:")
for i, headline in enumerate(headlines):
    print(f"{headline} -> Cluster {kmeans.labels_[i]}")

w1 = input("\nEnter first word: ").strip()
w2 = input("Enter second word: ").strip()
s1 = wn.synsets(w1)
s2 = wn.synsets(w2)
if s1 and s2:
    sim = s1[0].path_similarity(s2[0])
    print(f"\nWordNet Similarity: {sim}")
else:
    print("\nSimilarity not found")