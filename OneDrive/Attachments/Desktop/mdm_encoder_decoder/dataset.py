"""
Dataset utilities for Seq2Seq translation task.
Uses English-to-French sentence pairs for demonstration.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# Small curated English-French parallel corpus
RAW_DATA = [
    ("i am a student", "je suis un etudiant"),
    ("he is a teacher", "il est un professeur"),
    ("she is happy", "elle est heureuse"),
    ("we are friends", "nous sommes amis"),
    ("they are here", "ils sont ici"),
    ("the cat is small", "le chat est petit"),
    ("the dog is big", "le chien est grand"),
    ("i love music", "j aime la musique"),
    ("she reads a book", "elle lit un livre"),
    ("he plays football", "il joue au football"),
    ("we eat bread", "nous mangeons du pain"),
    ("they drink water", "ils boivent de l eau"),
    ("the sun is bright", "le soleil est brillant"),
    ("the moon is beautiful", "la lune est belle"),
    ("i have a car", "j ai une voiture"),
    ("she has a house", "elle a une maison"),
    ("he wants to eat", "il veut manger"),
    ("we need help", "nous avons besoin d aide"),
    ("they like to swim", "ils aiment nager"),
    ("the bird can fly", "l oiseau peut voler"),
    ("i go to school", "je vais a l ecole"),
    ("she works at home", "elle travaille a la maison"),
    ("he runs every day", "il court tous les jours"),
    ("we study mathematics", "nous etudions les mathematiques"),
    ("they speak english", "ils parlent anglais"),
    ("the baby is sleeping", "le bebe dort"),
    ("the flower is red", "la fleur est rouge"),
    ("i write a letter", "j ecris une lettre"),
    ("she sings a song", "elle chante une chanson"),
    ("he drives a car", "il conduit une voiture"),
    ("we watch television", "nous regardons la television"),
    ("they play music", "ils jouent de la musique"),
    ("the water is cold", "l eau est froide"),
    ("the food is good", "la nourriture est bonne"),
    ("i am very tired", "je suis tres fatigue"),
    ("she is very smart", "elle est tres intelligente"),
    ("he is my brother", "il est mon frere"),
    ("we are at school", "nous sommes a l ecole"),
    ("they are very kind", "ils sont tres gentils"),
    ("the sky is blue", "le ciel est bleu"),
    ("i like this book", "j aime ce livre"),
    ("she wants a cat", "elle veut un chat"),
    ("he needs a pen", "il a besoin d un stylo"),
    ("we have two dogs", "nous avons deux chiens"),
    ("they see the mountain", "ils voient la montagne"),
    ("the tree is tall", "l arbre est grand"),
    ("the river is long", "la riviere est longue"),
    ("i can speak french", "je peux parler francais"),
    ("she can dance well", "elle peut bien danser"),
    ("he knows the answer", "il connait la reponse"),
    ("i am learning french", "j apprends le francais"),
    ("she is cooking dinner", "elle prepare le diner"),
    ("he is reading a newspaper", "il lit un journal"),
    ("we are going home", "nous rentrons a la maison"),
    ("they are playing outside", "ils jouent dehors"),
    ("the children are happy", "les enfants sont heureux"),
    ("the weather is nice today", "le temps est beau aujourd hui"),
    ("i want to travel", "je veux voyager"),
    ("she likes chocolate", "elle aime le chocolat"),
    ("he eats an apple", "il mange une pomme"),
    ("we live in a city", "nous vivons dans une ville"),
    ("they work together", "ils travaillent ensemble"),
    ("the house is white", "la maison est blanche"),
    ("the car is fast", "la voiture est rapide"),
    ("i drink coffee every morning", "je bois du cafe chaque matin"),
    ("she teaches mathematics at school", "elle enseigne les mathematiques a l ecole"),
    ("he plays guitar very well", "il joue de la guitare tres bien"),
    ("we visit our grandparents often", "nous visitons nos grands parents souvent"),
    ("they travel around the world", "ils voyagent autour du monde"),
    ("the garden has many flowers", "le jardin a beaucoup de fleurs"),
    ("the museum is very old", "le musee est tres vieux"),
    ("i enjoy reading books at night", "j aime lire des livres la nuit"),
    ("she always helps her friends", "elle aide toujours ses amis"),
    ("he studies hard for exams", "il etudie dur pour les examens"),
    ("we celebrate birthdays with cake", "nous celebrons les anniversaires avec un gateau"),
    ("they learn new languages quickly", "ils apprennent de nouvelles langues rapidement"),
    ("the movie was very interesting", "le film etait tres interessant"),
    ("the restaurant serves good food", "le restaurant sert de la bonne nourriture"),
    ("i remember my first day at school", "je me souviens de mon premier jour a l ecole"),
    ("she paints beautiful pictures of nature", "elle peint de beaux tableaux de la nature"),
    ("good morning", "bonjour"),
    ("good night", "bonne nuit"),
    ("thank you very much", "merci beaucoup"),
    ("see you tomorrow", "a demain"),
    ("how are you", "comment allez vous"),
    ("i am fine", "je vais bien"),
    ("what is your name", "comment vous appelez vous"),
    ("where do you live", "ou habitez vous"),
    ("i like to read", "j aime lire"),
    ("the book is on the table", "le livre est sur la table"),
    ("she is my best friend", "elle est ma meilleure amie"),
    ("he works in a hospital", "il travaille dans un hopital"),
    ("we go to the park", "nous allons au parc"),
    ("the stars shine at night", "les etoiles brillent la nuit"),
    ("i have three brothers", "j ai trois freres"),
    ("she wears a blue dress", "elle porte une robe bleue"),
    ("he bought a new phone", "il a achete un nouveau telephone"),
    ("we finished our homework", "nous avons fini nos devoirs"),
    ("they arrived yesterday", "ils sont arrives hier"),
    ("the train is late", "le train est en retard"),
    ("i will come back soon", "je reviendrai bientot"),
    ("she promised to help me", "elle a promis de m aider"),
]

# Duplicate with small variations to increase dataset size
EXTRA = [
    ("i am happy", "je suis heureux"),
    ("he is sad", "il est triste"),
    ("she is tall", "elle est grande"),
    ("we are students", "nous sommes etudiants"),
    ("they are teachers", "ils sont professeurs"),
    ("the cat sleeps", "le chat dort"),
    ("the dog runs", "le chien court"),
    ("i eat rice", "je mange du riz"),
    ("she drinks milk", "elle boit du lait"),
    ("he reads books", "il lit des livres"),
    ("we play games", "nous jouons a des jeux"),
    ("they sing songs", "ils chantent des chansons"),
    ("the night is dark", "la nuit est sombre"),
    ("the day is long", "la journee est longue"),
    ("i have a dog", "j ai un chien"),
    ("she has a cat", "elle a un chat"),
    ("he wants to sleep", "il veut dormir"),
    ("we need water", "nous avons besoin d eau"),
    ("they like to dance", "ils aiment danser"),
    ("the fish can swim", "le poisson peut nager"),
    ("i go to work", "je vais au travail"),
    ("she studies at university", "elle etudie a l universite"),
    ("he walks every morning", "il marche chaque matin"),
    ("we learn science", "nous apprenons les sciences"),
    ("they speak french", "ils parlent francais"),
    ("i am not tired", "je ne suis pas fatigue"),
    ("she is not here", "elle n est pas ici"),
    ("he does not understand", "il ne comprend pas"),
    ("we do not agree", "nous ne sommes pas d accord"),
    ("they did not come", "ils ne sont pas venus"),
]

RAW_DATA.extend(EXTRA)

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.word2idx[tok] = i
            self.idx2word[i] = tok

    def build(self, sentences):
        for sent in sentences:
            for word in sent.split():
                self.word_count[word] += 1
        idx = len(self.word2idx)
        for word in self.word_count:
            if word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, sentence, max_len=None):
        tokens = [self.word2idx.get(w, self.word2idx[UNK_TOKEN]) for w in sentence.split()]
        tokens = [self.word2idx[SOS_TOKEN]] + tokens + [self.word2idx[EOS_TOKEN]]
        if max_len:
            tokens = tokens[:max_len]
            tokens += [self.word2idx[PAD_TOKEN]] * (max_len - len(tokens))
        return tokens

    def decode(self, indices):
        words = []
        for idx in indices:
            w = self.idx2word.get(idx, UNK_TOKEN)
            if w == EOS_TOKEN:
                break
            if w not in (PAD_TOKEN, SOS_TOKEN):
                words.append(w)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, trg_vocab, max_len=20):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, trg = self.pairs[idx]
        src_enc = self.src_vocab.encode(src, self.max_len)
        trg_enc = self.trg_vocab.encode(trg, self.max_len)
        return torch.tensor(src_enc), torch.tensor(trg_enc)


def get_data(batch_size=32, max_len=20):
    """Prepare datasets and dataloaders."""
    src_sentences = [p[0] for p in RAW_DATA]
    trg_sentences = [p[1] for p in RAW_DATA]

    src_vocab = Vocabulary()
    trg_vocab = Vocabulary()
    src_vocab.build(src_sentences)
    trg_vocab.build(trg_sentences)

    # 80/20 split
    split = int(0.8 * len(RAW_DATA))
    train_pairs = RAW_DATA[:split]
    test_pairs = RAW_DATA[split:]

    train_ds = TranslationDataset(train_pairs, src_vocab, trg_vocab, max_len)
    test_ds = TranslationDataset(test_pairs, src_vocab, trg_vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, src_vocab, trg_vocab, train_pairs, test_pairs
