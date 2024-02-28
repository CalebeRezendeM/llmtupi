import re
from collections import Counter, defaultdict
from tqdm import tqdm
import re

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
    
    # Extract the final set of tokens and their frequencies
    tokens = dict()
    for word, count in vocab.items():
        for token in word.split():
            if token in tokens:
                tokens[token] += count
            else:
                tokens[token] = count
    
    tokens[' '] = vocab.get(' ', 0)
    return tokens

normalize_tupi = lambda x: re.sub('\s+', ' ', x.replace('-','').replace('.', '').replace('?', '').replace(',', '')).lower()
# file 'docs/citations.csv' using the csv module and only select the rows of column `Tupi`
import csv
tupi_text = ''
tupi_rows = []
with open('docs/citations.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        tupi_text += row['Tupi'] + ' '
        tupi_rows.append(row['Tupi'])
tupi_text = normalize_tupi(tupi_text)

tokens = bpe_tokenize(tupi_text, 100)
print(tokens)
# print a statistical summary of the tokens
print(f"Number of tokens: {len(tokens)}")
# Validate the tokens created against the original text
for token in tokens:
    assert token in tupi_text

def tokenize_sentence(sentence_raw, tokens, layer=0):
    sentence = normalize_tupi(sentence_raw)
    # print(f"Sentence: {sentence}, Layer: {layer}")
    tokenized_sentence = []
    while sentence:
        curr_word = sentence.split(' ')[0]
        matches = [x for x in tokens.items() if x[0] in curr_word]
        if matches:
            for token, _ in sorted(matches, key=lambda x:len(x[0]), reverse=True):
                loc_tokens = [token]
                ind = curr_word.index(token)
                left, right = curr_word[:ind], curr_word[ind+len(token):]
                if left:
                    loc_tokens = tokenize_sentence(left, tokens, layer+1) + loc_tokens
                if right:
                    loc_tokens += tokenize_sentence(right, tokens, layer+1)
                tokenized_sentence += loc_tokens
                if ' ' in sentence:
                    sentence = sentence[sentence.index(' '):].strip()
                else:
                    sentence = ''
                # print(f"\tToken: {token}, Sentence: {sentence}")
                break
        else:
            token = curr_word
            tokenized_sentence.append(token)
            sentence = sentence[len(token):].strip()
        # print(f"\tToken: {token}, Sentence: {sentence}")
    return tokenized_sentence

# Tokenize the entire Tupi text limit 15 and print
for i, row in enumerate(tupi_rows[:15]):
    row = normalize_tupi(row)
    tokenized_sentence = tokenize_sentence(row, tokens)
    print(f"Original:\t {row}\nTokenized:\t {'|'.join(tokenized_sentence)}\n")

# write tupi_rows lines to a file
with open('docs/tupi_text.txt', 'w') as file:
    for row in tupi_rows:
        file.write(normalize_tupi(row) + '\n')

# import 'docs/dict-conjugated.json' using the json module
import json
with open('docs/dict-conjugated.json', 'r') as file:
    dicc = json.load(file)


dicc_dict = {i: v for i, v in enumerate(dicc)}
tupi_only = []
include = False
adjectives = []
ban = [
    "NOTA",
    "Daí",
    "De",
    "OBSERVAÇÃO",
    "Daí,",
    "aba",
    "-ab",
    "abatiputá",
    "-agûama",
    "a'ebé",
    "agûaîxima",
    "agûaragûasu",
    "agûy",
    "ambûer",
    "apyrĩ",
    "ambype",
    "gûaîá",
    "eno-",
    "îabotimirĩ",
    "îapĩ",
    "Maíra",
    "memetipó",
    "moro-",
    "muresi",
    "pyru'ã",
    "POROROCA",
    "sybyamumbyaré",
    "Muitos",
    "Há",
    "O",
    "Cardim,",
]
for i, vbt in dicc_dict.items():
    if vbt["f"] == "ã":
        include = True
    if include and vbt["f"] not in ban and "adj.: " not in vbt["d"]:
        vbt["id"] = i
        tupi_only.append(vbt)
    if vbt["f"] == "'yura":
        include = False

adj_raws = [
    (
        x["f"].replace("(e)", "e").replace("teînhẽa", "tenhẽa"),
        x["d"]
        .split("adj.: ")[1]
        .split(") ")[0]
        .split("):")[0]
        .split(" ")[0]
        .replace(",", "")
        .replace("(e)", "e")
        .replace("ygapenung", "yapenung"),
        x["o"],
        x["d"],
        i,
    )
    for i, x in dicc_dict.items()
    if "adj.: " in x["d"]
]
for first_word, optional_number, definition, vid in {
    (x[1], x[2], x[3], x[4]) for x in adj_raws
}:
    tupi_only.append(
        {
            "f": first_word,
            "o": optional_number,
            "d": definition,
            "id": vid,
        }
    )

wordlist = {normalize_tupi(v["f"]).strip() for v in tupi_only}
# wordlist = list(wordlist) + list({normalize_tupi(v.strip()).strip() for v in tupi_text.replace('\n', ' ').split(' ')})

# identift the index of the first numeric character in a string
def first_numeric_index(s):
    for i, c in enumerate(s):
        if c.isdigit():
            return i
    return False
navarro =  "p pû pî b t s sû k kû ' m mû n r nh ng mb mbû nd ndû ng ngû gû g û î ŷ a á e é i í y ý o ó u ú ã ẽ ĩ ỹ õ ũ x".split(
            " ")
def is_made_of_substrings(s, substrings):
    if s == '':
        return True
    else:
        for substring in substrings:
            if s.startswith(substring):
                if is_made_of_substrings(s[len(substring):], substrings):
                    return True
        return False
#write a function which checks if a string contains characters not within navarro
def is_navarro(s):
    return is_made_of_substrings(s, navarro)

wordlist = {v[:first_numeric_index(v)] if first_numeric_index(v) else v for v in wordlist}
wordlist = {v.strip() for v in wordlist if is_navarro(v) and  v != ""}

import sentencepiece as spm
spm.SentencePieceTrainer.train(input='docs/tupi_text.txt'
                               , model_prefix='m'
                               , vocab_size=7362
                               , model_type='bpe'
                               , user_defined_symbols=list(wordlist))
sp = spm.SentencePieceProcessor(model_file='m.model')

for i, row in enumerate(tupi_rows[:150]):
    row = normalize_tupi(row)
    tokenized_sentence = sp.encode(row, out_type=str)
    print(f"Original:\t {row}\nTokenized:\t {'|'.join(tokenized_sentence)}\n")
