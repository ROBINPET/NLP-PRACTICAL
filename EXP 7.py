import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist

for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

tweet = input("Enter a tweet: ")
tokens = nltk.word_tokenize(tweet.lower())
print("\nTokens:")
print(tokens)

unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))
print("\nUnigrams:")
print(unigrams)
print("\nBigrams:")
print(bigrams)
print("\nTrigrams:")
print(trigrams)

fd = FreqDist(tokens)
print("\nWord Frequencies:")
for word, freq in fd.items():
    print(word, ":", freq)

print("\nHMM Prediction (Sample)")
print("AI-> NOUN")
print("improves-> VERB")
print("technology-> NOUN")