# -*- coding: utf-8 -*-
#!/usr/bin/env python

import sys
import re
import operator
from gensim import corpora, models, similarities
import pickle


if __name__ == '__main__':
  if len(sys.argv) >= 2:
    vector_label = unicode(sys.argv[1], "utf-8")
    num_topics_lsa = int(sys.argv[2]) if len(sys.argv) >= 3 else 100
    num_topics_lda = int(sys.argv[3]) if len(sys.argv) >= 4 else 100
    limit = int(sys.argv[4]) if len(sys.argv) >= 5 else 10
    limit_topics = int(sys.argv[5]) if len(sys.argv) >= 6 else 10

    lsa_model = models.lsimodel.LsiModel.load('/tmp/lsa_model_' + str(num_topics_lsa) + '.lsa')
    lda_model = models.ldamodel.LdaModel.load('/tmp/lda_model_' + str(num_topics_lda) + '.lda')
    output_labels = open('/tmp/labels.pkl', 'rb')
    output_documents = open('/tmp/documents.pkl', 'rb')
    output_tfidf = open('/tmp/tfidf.pkl', 'rb')
    output_corpus_lsa = open('/tmp/corpus_lsa_' + str(num_topics_lsa) + '.pkl', 'rb')
    output_corpus_lda = open('/tmp/corpus_lda_' + str(num_topics_lda) + '.pkl', 'rb')

    labels = pickle.load(output_labels)
    documents = pickle.load(output_documents)
    tfidf = pickle.load(output_tfidf)
    corpus_lsa = pickle.load(output_corpus_lsa)
    corpus_lda = pickle.load(output_corpus_lda)
    output_labels.close()
    output_documents.close()
    output_tfidf.close()
    output_corpus_lsa.close()
    output_corpus_lda.close()

    print "models loaded"

    lsa_index = similarities.SparseMatrixSimilarity(lsa_model[corpus_lsa], num_features=num_topics_lsa)
    lda_index = similarities.SparseMatrixSimilarity(lda_model[corpus_lda], num_features=num_topics_lda)

    print "similarities matrices generated"

    vector_id = labels.index(vector_label)
    vector = documents[vector_id]
    vector_tfidf = tfidf[vector]

    print "LSA rank:"
    sims_lsa = lsa_index[ lsa_model[vector_tfidf] ]

    topics_set = {}
    for i in sorted(enumerate(sims_lsa), key=lambda item: -item[1])[:limit]:
      topics = [item[0] for item in sorted( lsa_model[tfidf[documents[i[0]]]], key = lambda item: -item[1])[:limit_topics] ]
      for topic in topics:
        if topic not in topics_set:
          topics_set[topic] = 1.
        else:
          topics_set[topic] += 1.
      print labels[i[0]], " : ", i[1], " : topics:", topics
    print "Top topics:"
    for i in sorted(topics_set.iteritems(), key=lambda x:-x[1])[:limit_topics]:
      print "topic[", i[0], "] = ", lsa_model.print_topic(i[0])

    topics_set = {}
    print "LDA rank:"
    sims_lda = lda_index[ lda_model[vector_tfidf] ]
    for i in sorted(enumerate(sims_lda), key=lambda item: -item[1])[:limit]:
      topics = [item[0] for item in sorted( lda_model[tfidf[documents[i[0]]]], key = lambda item: -item[1])[:limit_topics] ]
      for topic in topics:
        if topic not in topics_set:
          topics_set[topic] = 1.
        else:
          topics_set[topic] += 1.
      print labels[i[0]], " : ", i[1], " : topics:", topics
    print "Top topics:"
    for i in sorted(topics_set.iteritems(), key=lambda x:-x[1])[:limit_topics]:
      print "topic[", i[0], "] = ", lda_model.print_topic(i[0])

  else:
    print("python -m search_vectors [vector_label] [number of topics for lsa=100] [number of topics for lda=100]")

