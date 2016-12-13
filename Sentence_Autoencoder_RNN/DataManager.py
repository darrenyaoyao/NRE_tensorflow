import numpy as np
import nltk
import itertools


class DataManager:
    def __init__(self):
        self.limit = 30
        self.unknown_token = "UNKNOWN_TOKEN"
        self.vocabulary_size = 8000
        self.maxlength = 0
        self.PositionMinE1 = self.PositionMaxE1 = self.PositionTotalE1 = self.PositionMinE2 = self.PositionMaxE2 = self.PositionTotalE2 = 0
        self.wordvectors = {}
        self.wordlist = []
        self.wordMapping = {}
        self.wordvector_dim = 0
        self.wordtotal = 0
        self.relationtotal = 0
        self.relationMapping = {}
        self.relationlist = []
        self.trainlist = []
        self.trainlist_word = []
        self.testlist = []
        self.load_word2vec()
        self.load_relation()

    def load_word2vec(self):
        #load word2vec from file
        wordvector = list(open("../data/vector1.txt", "r").readlines())
        wordvector = [s.split() for s in wordvector]
        self.wordvector_dim = len(wordvector[0])-1
        self.wordlist.append("UNK")
        self.wordMapping["UNK"] = 0
        self.wordvectors["UNK"] = np.zeros(self.wordvector_dim)
        index = 1
        for vec in wordvector:
            a = np.zeros(self.wordvector_dim)
            for i in range(self.wordvector_dim):
                a[i] = float(vec[i+1])
            self.wordvectors[vec[0]] = a
            self.wordlist.append(vec[0])
            self.wordMapping[vec[0]] = index
            index += 1

        print("WordTotal=\t", len(self.wordvectors))
        print("Word dimension=\t", self.wordvector_dim)
        self.wordtotal = len(self.wordvectors)+1

    def load_relation(self):
        #load relation from file
        relation_data = list(open("../data/RE/relation2id.txt").readlines())
        relation_data = [s.split() for s in relation_data]
        for relation in relation_data:
            self.relationMapping[relation[0]] = float(relation[1])
        self.relationtotal = len(self.relationMapping)

    def load_training_data(self, include_NA=False):
        #load training data from file
        training_data = list(open("../data/RE/train.txt").readlines())
        training_data = [s.split() for s in training_data]
        for data in training_data:
            e1 = data[0]
            e2 = data[1]
            head_s = data[2]
            if head_s not in self.wordMapping:
                head = 0
            else:
                head = self.wordMapping[head_s]
            tail_s = data[3]
            if tail_s not in self.wordMapping:
                tail = 0
            else:
                tail = self.wordMapping[tail_s]
            relation_s = data[4]
            relation = self.relationMapping[relation_s]
            if relation == 0 and not include_NA:
                continue
            lefnum = rignum = 0
            seq_vector = []
            seq_word = []
            for i in range(5, len(data)-1):
                if data[i] not in self.wordMapping:
                    gg = self.wordvectors["UNK"]
                else:
                    gg = self.wordvectors[data[i]]
                if data[i] == head_s:
                    lefnum = i-5
                    seq_word.append("<entity>")
                elif data[i] == tail_s:
                    rignum = i-5
                    seq_word.append("<entity>")
                else:
                    seq_word.append(data[i])
                seq_vector.append(gg)
            seq_word.append("<end>")
            r = np.zeros(self.relationtotal)
            r[int(relation)] = 1.0
            self.relationlist.append(r)
            for i in range(len(seq_vector)):
                seq_vector[i] = np.insert(seq_vector[i], len(seq_vector[i]), i-lefnum)
                seq_vector[i] = np.insert(seq_vector[i], len(seq_vector[i]), i-rignum)
            self.trainlist.append(seq_vector)
            self.trainlist_word.append(seq_word)
        self.word2num()
        return self.trainlist, np.asarray(self.relationlist), self.trainlist_num

    def load_testing_data(self):
        #load training data from file
        testing_data = list(open("../data/RE/test1.txt").readlines())
        testing_data = [s.split() for s in testing_data]
        for data in testing_data:
            e1 = data[0]
            e2 = data[1]
            head_s = data[2]
            if head_s not in self.wordMapping:
                head = 0
            else:
                head = self.wordMapping[head_s]
            tail_s = data[3]
            if tail_s not in self.wordMapping:
                tail = 0
            else:
                tail = self.wordMapping[tail_s]
            relation_s = data[4]
            relation = self.relationMapping[relation_s]
            if relation == 0:
                continue
            lefnum = rignum = 0
            sequence = []
            for i in range(5, len(data)-1):
                if data[i] not in self.wordMapping:
                    gg = self.wordvectors["UNK"]
                else:
                    gg = self.wordvectors[data[i]]
                if data[i] == head_s:
                    lefnum = i-5
                if data[i] == tail_s:
                    rignum = i-5
                sequence.append(gg)
            self.relationlist.append(relation)
            for i in range(len(sequence)):
                sequence[i] = np.insert(sequence[i], len(sequence[i]), i-lefnum)
                sequence[i] = np.insert(sequence[i], len(sequence[i]), i-rignum)
            self.testlist.append(sequence)
        return self.testlist, np.asarray(self.relationlist)

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.asarray(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
            #Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                if start_index == end_index:
                    continue
                else:
                    yield shuffled_data[start_index:end_index]

    def seq2seq_batch_iter(self, x, y, y_label, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        #data = np.asarray(data)
        data_size = len(x)
        num_batches_per_epoch = int(len(x)/batch_size) + 1
        for epoch in range(num_epochs):
            #Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                x = x[shuffle_indices]
                y = y[shuffle_indices]
                y_label = y_label[shuffle_indices]
            shuffled_data = list(zip(x, y, y_label))
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                if start_index == end_index:
                    continue
                else:
                    yield shuffled_data[start_index:end_index]

    def word2num(self):
        """
        Translate most common words to number
        """
        word_freq = nltk.FreqDist(itertools.chain(*self.trainlist_word))
        self.endnum = word_freq['<end>']
        vocab = word_freq.most_common(self.vocabulary_size-2)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(self.unknown_token)
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

        #Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(self.trainlist_word):
            self.trainlist_word[i] = [w if w in word_to_index else self.unknown_token for w in sent]

        #Create the number training data
        self.trainlist_num = np.array([[word_to_index[w]+1 for w in sent] for sent in self.trainlist_word])
