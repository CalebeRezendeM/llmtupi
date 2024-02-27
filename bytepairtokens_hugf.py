import re
from collections import Counter, defaultdict
from tqdm import tqdm
import re
from transformers import AutoTokenizer

# Completey bare bytpair tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def get_vocab(text):
    """Extracts vocabulary and counts from the text."""
    words = re.findall(r"[\s.?,]+([^\s.?,]+)[\s.?,]+", text)
    vocab = Counter(words)
    return {word: count for word, count in vocab.items()}

def get_stats(vocab):
    """Computes pairs of consecutive symbols in the vocabulary."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Merges the most frequent pair in the vocabulary."""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def bpe_tokenize(text, num_merges=None):
    """Performs BPE tokenization on the input text."""
    vocab = get_vocab(text)
    vocab = {' '.join(word): count for word, count in vocab.items()}
    i=0
    # Create a tqdm manual object which is titled 'Merging tokens' and has a total of num_merges if not none or total vocab to get to if none
    pbar = tqdm(total=num_merges if num_merges else len(vocab), desc='Merging tokens')

    while True:
        if num_merges:
            i+=1
        pbar.update(1)
        if num_merges and i >= num_merges:
            break
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    
    # Extract the final set of tokens
    tokens = set()
    for word in vocab:
        tokens.update(word.split())
    
    tokens.update([' '])
    return tokens

normalize_tupi = lambda x: re.sub('\s+', ' ', x.replace('-','').replace('.', '').replace('?', '').replace(',', '')).lower()
# file 'docs/citations.csv' using the csv module and only select the rows of column `Tupi`
import csv
tupi_text = ''
corpus = []
with open('docs/citations.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        tupi_text += row['Tupi'] + ' '
        corpus.append(normalize_tupi(row['Tupi']))
tupi_text = normalize_tupi(tupi_text)

from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)

alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

print(alphabet)

vocab = [" "] + alphabet.copy()
splits = {word: [c for c in word] for word in word_freqs.keys()}
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs
pair_freqs = compute_pair_freqs(splits)

for i, key in enumerate(pair_freqs.keys()):
    print(f"{key}: {pair_freqs[key]}")
    if i >= 5:
        break

def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits
vocab_size = 50
merges = dict()
while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])

print(merges)
print(vocab)
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])
print(tokenize(corpus[0]))