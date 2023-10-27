import csv
import pickle
import tensorflow as tf
import tqdm
import re
from collections import defaultdict

class TokenizerEmbedder():
    PAD = 0

    def __init__(self, bpe_file: str, vectors_file: str, tokens_file: str):
        self._embedder_table = []
        self._tokens = defaultdict(lambda: 1)

        with open(bpe_file, 'rb') as fd:
            self._bpe = pickle.load(fd)

        with open(vectors_file, encoding='utf-8') as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                self._embedder_table.append([float(v) for v in row])

        with open(tokens_file, encoding='utf-8') as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar=None)
            for idx, row in enumerate(rd):
                self._tokens[row[0]] = idx
        self._inverse_tokens = {index: token for token, index in self._tokens.items()}
                
    def tokenize(self, corpus: list[str]):
        corpus = [sentance.lower() for sentance in corpus]
        words = []
        for sentance in corpus:
            words += [' '.join(list(word)) + ' </w>' for word in sentance.split()]
            words += ['[END]']
        str_tensor = tf.constant(words)
        for pair in tqdm.tqdm(self._bpe):
            pattern = r'( |^)' + re.escape(' '.join(pair)) + r'( |$)'
            replacement = r'\1' + ''.join(pair) + r'\2'
            str_tensor = tf.strings.regex_replace(str_tensor, pattern, replacement)
        tokens: list[list[int]] = []
        tokens_sentance = []
        for word in str_tensor:
            word = word.numpy().decode('utf-8')
            bpe_tokens = word.split()
            tokenization = [self._tokens[token] for token in bpe_tokens]
            tokens_sentance += tokenization
            if word == '[END]':
                tokens.append(tokens_sentance)
                tokens_sentance = []
        return tokens

    def embed(self, tokens: list[list[int]]) -> list[list[list[float]]]:
        embedding = []
        for sentance in tokens:
            embeddings_sentance = []
            for token in sentance:
                embeddings_sentance.append(self._embedder_table[token])
            embedding.append(embeddings_sentance)
        return embedding
    
    def __call__(self, corpus: list[str]):
        tokens = self.tokenize(corpus)
        embeddings = self.embed(tokens)
        return embeddings

    def __getitem__(self, key):
        if isinstance(key, int):
            return (self._inverse_tokens[key], self._embedder_table[key])
        elif isinstance(key, str):
            return self._tokens[key]
        else:
            raise TypeError(f'{key} is not a valid indexing type.')
            

if __name__ == '__main__':
    te = TokenizerEmbedder('bpe.pckl', 'vectors.tsv', 'metadata.tsv')
    st = 'hej världen! Vad soligt det är idag :)!'
    tokens = te.tokenize(['hej världen! Vad soligt det är idag :)!'])
    print(st)
    for token in tokens[0]:
        print(te[token][0], end=' ')
    print()
    print(*tokens[0])
    print(te.embed(tokens))
    # print()
    # for token in tokens[1]:
    #     print(te[token][0], end=' ')
