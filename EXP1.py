#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required data (harmless if already present)
nltk.download('punkt')
nltk.download('wordnet')

# Get input from CLI args if provided, else prompt / read stdin
if len(sys.argv) > 1:
    text = " ".join(sys.argv[1:])
else:
    try:
        text = input("Enter a paragraph: ")
    except EOFError:
        text = ""

# Sentence tokenization with fallback
try:
    sentences = sent_tokenize(text)
except LookupError:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

# Word tokenization with fallback
try:
    tokens = word_tokenize(text)
except LookupError:
    tokens = text.split()

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

# Box printing helper
def print_box(lines):
    if not lines:
        return
    width = max(len(line) for line in lines)
    sep = "+" + "-" * (width + 2) + "+"
    print(sep)
    for line in lines:
        print("| " + line.ljust(width) + " |")
    print(sep)

# Prepare per-sentence word lists and POS tags (if available)
try:
    nltk.download('averaged_perceptron_tagger')
    has_tagger = True
except Exception:
    has_tagger = False

sentence_word_info = []
for s in sentences:
    try:
        s_tokens = word_tokenize(s)
    except LookupError:
        s_tokens = s.split()
    s_pos = []
    if has_tagger:
        try:
            s_pos = nltk.pos_tag(s_tokens)
        except Exception:
            s_pos = [(w, '') for w in s_tokens]
    else:
        s_pos = [(w, '') for w in s_tokens]
    sentence_word_info.append(s_pos)

lines = []
lines.append("Original Text:")
lines.append(text)
lines.append("")
for i, s in enumerate(sentences, start=1):
    lines.append(f"Sentence {i}:")
    lines.append(s)
    for j, (w, tag) in enumerate(sentence_word_info[i-1], start=1):
        if tag:
            lines.append(f"{j}. {w} ({tag})")
        else:
            lines.append(f"{j}. {w}")
    lines.append("")

lines.append("Comparison:")
lines.append("Stemming reduces words to root forms, which may not be meaningful.")
lines.append("Lemmatization converts words to meaningful base forms.")

print_box(lines)