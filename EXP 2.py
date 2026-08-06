import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import argparse
import sys


def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)


def main():
    ensure_nltk_data()
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='Input sentence', default=None)
    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            text = input("Enter a sentence:")

    tokens = word_tokenize(text)
    tagged_words = pos_tag(tokens)

    print("\nTokens:")
    print(tokens)

    print("\nPOS Tags:")
    for word, tag in tagged_words:
        print(word, "->", tag)

    # Simple tag meanings
    print("\nTag Meanings:")
    print("NN -> Noun")
    print("VB -> Verb")
    print("JJ -> Adjective")
    print("RB -> Adverb")
    print("PRP -> Pronoun")
    print("DT -> Determiner")

    # Count tagged words
    print("\nTotal Words:", len(tokens))


if __name__ == '__main__':
    main()