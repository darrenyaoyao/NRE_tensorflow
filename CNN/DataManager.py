import numpy as np

class DataManager:
    def __init__(self):
        self.limit = 30
        self.maxlength = 0
        self.PositionMinE1 = self.PositionMaxE1 = self.PositionTotalE1 = self.PositionMinE2 = self.PositionMaxE2 = self.PositionTotalE2 = 0
        self.wordvectors = {}
        self.wordlist = []
        self.wordMapping = {}
        self.wordvector_dim = 0
        self.wordtotal = 0
        self.relationtotal = 0
        self.relationMapping = {}
        self.headlist = []
        self.taillist = []
        self.relationlist = []
        self.trainlength = []
        self.trainlist = []
        self.trainpositionE1 = []
        self.trainpositionE2 = []
        self.testlength = []
        self.testlist = []
        self.testpositionE1 = []
        self.testpositionE2 = []
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

    def load_training_data(self):
        #load training data from file
        training_data = list(open("../data/RE/train1.txt").readlines())
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
            self.headlist.append(head)
            self.taillist.append(tail)
            r = np.zeros(self.relationtotal)
            r[int(relation)] = 1.0
            self.relationlist.append(r)
            for i in range(len(sequence)):
                sequence[i] = np.insert(sequence[i], len(sequence[i]), lefnum)
                sequence[i] = np.insert(sequence[i], len(sequence[i]), rignum)
            self.trainlist.append(sequence)
        return self.trainlist, np.asarray(self.relationlist)

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
            length = len(data)-6
            self.headlist.append(head)
            self.taillist.append(tail)
            self.relationlist.append(relation)
            for i in range(len(sequence)):
                sequence[i] = np.insert(sequence[i], len(sequence[i]), lefnum)
                sequence[i] = np.insert(sequence[i], len(sequence[i]), rignum)
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
                yield shuffled_data[start_index:end_index]
