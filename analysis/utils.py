# Utility function for NER extraction
import nltk


def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names

def ner_tag_line(line):
    pos_tagged_sentences = pos_tag_sentences(line)
    chunked_sentences = nltk.ne_chunk_sents(pos_tagged_sentences, binary=True)

    return chunked_sentences

def pos_tag_sentences(line):
    tokenized_sentences = tokenize_sentences(line)
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    return tagged_sentences


def tokenize_sentences(line):
    sentences = nltk.sent_tokenize(line)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    return tokenized_sentences