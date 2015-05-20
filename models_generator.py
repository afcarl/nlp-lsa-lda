# -*- coding: utf-8 -*-
#!/usr/bin/env python

import sys
import re
import operator
from gensim import corpora, models, similarities
import pickle

def create_synonyms_dictionary(synonyms_file):
  synonyms = {}
  with open(synonyms_file) as f:
    for line in f:
      words = unicode(line, "utf-8").split(', ')
      for word in words[1:]:
        synonyms[word] = words[0]
  return synonyms

def vectorize(corpus_file, synonyms):
  with open(corpus_file) as f:
    data = unicode(f.read(), "utf-8")
  # data = re.findall(ur"(#[\d]+)([^#]*)#[\d]+", data, re.DOTALL)
  data = re.findall(ur"(#[\d]+)([^#]*)(?#[\d]+)", data, re.DOTALL)
  labels = []
  documents = []

  for item in data:
    labels.append(item[0])
    value = [word.lower() for word in re.findall(ur"[a-zA-ZżółćęśąźńŻÓŁĆĘŚĄŹŃ]+", item[1])]
    value = [synonyms[word] if word in synonyms else word for word in value]
    documents.append(value)

  return labels, documents

def wordcount(documents, document_frequency=False):
  words = {}
  for document in documents:
    checked = {}
    for word in document:
      if document_frequency == False or word not in checked:
        if word not in words:
          words[word] = 1.
        else:
          words[word] += 1.
      checked[word] = True
  return words

def filter_words_in_documents(documents, hapax_legomena, too_frequent_temrs):
  for i in xrange(len(documents)):
    documents[i] = [word for word in documents[i] if word not in hapax_legomena and word not in too_frequent_temrs]
  return documents

if __name__ == '__main__':
  if len(sys.argv) >= 3:
    corpus_file = sys.argv[1]
    synonyms_file = sys.argv[2]

    num_topics_lsa = int(sys.argv[3]) if len(sys.argv) >= 4 else 100
    num_topics_lda = int(sys.argv[4]) if len(sys.argv) >= 5 else 100

    print "Number of topics for LSA alrotithm: ", num_topics_lsa
    print "Number of topics for LDA alrotithm: ", num_topics_lda

    synonyms = create_synonyms_dictionary(synonyms_file)
    labels, documents = vectorize(corpus_file, synonyms)

    print len(labels)

    words = wordcount(documents)
    hapax_legomena = {k : v for k, v in words.iteritems() if v == 1.}
    df = wordcount(documents, document_frequency=True)
    limit = len(documents)*0.7
    too_frequent_terms = {k : v for k, v in words.iteritems() if v > limit}

    print "all words: ", len(words)
    print "all documents: ", len(labels)
    print "hapax legomena: ", len(hapax_legomena)
    print "too frequent terms: ", len(too_frequent_terms)

    documents = filter_words_in_documents(documents, hapax_legomena, too_frequent_terms)

    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(document) for document in documents]

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    print "tfidf computed"

    lsa_model = models.lsimodel.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics_lsa)
    corpus_lsa = lsa_model[corpus_tfidf]

    print "lsa model computed"

    lda_model = models.ldamodel.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics_lda)
    corpus_lda = lda_model[corpus_tfidf]

    print "lda model computed"

    lsa_model.save('/tmp/lsa_model_' + str(num_topics_lsa) + '.lsa')
    lda_model.save('/tmp/lda_model_' + str(num_topics_lda) + '.lda')
    output_labels = open('/tmp/labels.pkl', 'wb')
    output_documents = open('/tmp/documents.pkl', 'wb')
    output_tfidf = open('/tmp/tfidf.pkl', 'wb')
    output_corpus_lsa = open('/tmp/corpus_lsa_' + str(num_topics_lsa) + '.pkl', 'wb')
    output_corpus_lda = open('/tmp/corpus_lda_' + str(num_topics_lda) + '.pkl', 'wb')
    pickle.dump(labels, output_labels)
    pickle.dump(corpus, output_documents)
    pickle.dump(tfidf, output_tfidf)
    pickle.dump(corpus_lsa, output_corpus_lsa)
    pickle.dump(corpus_lda, output_corpus_lda)
    output_labels.close()
    output_tfidf.close()
    output_documents.close()
    output_corpus_lsa.close()
    output_corpus_lda.close()
    print "models saved"

  else:
    print("python -p models_generator [corpus] [synonyms] [number of topics for lsa=100] [number of topics for lda=100]")
