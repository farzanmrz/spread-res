class Vocab:

    """
    Class that handles mapping to and from
    words to vocab indices.
    """

    # Define the init function
    def __init__( self, target = False, space = True, case = 'lower'):
        """
        Initializes the Vocab class.

        Parameters:
        target (bool): If True, the vocabulary is for target words and does not include special tokens.
        case (str): Case to be used to determine whether vocab is uppercase, lowercase or both
        space (bool): If True, spaces are included in the vocabulary.
        """
        self._target = target
        self._case = case
        self._space = space

        # Adjust special tokens as per case for upper and lower/both
        if self._case == 'lower':
            self.UNK = '<unk>'
            self.PAD = '<pad>'
            self.CLS = '<cls>'
            self.EOS = '<eos>'
        else:
            self.UNK = '<UNK>'
            self.PAD = '<PAD>'
            self.CLS = '<CLS>'
            self.EOS = '<EOS>'

        # Dictionary mapping words to indices, blank if target is True else made with special tokens.
        self._word2idx = { } if self._target else { self.UNK: 0, self.PAD: 1, self.CLS: 2, self.EOS: 3 }

        # Dictionary mapping indices to words, reverses lookup for the vocab.
        self._idx2word = { v: k for k, v in self._word2idx.items() }


    def train( self, tokens ):
        """
        Trains the vocabulary on a list of tokens.

        Parameters:
        tokens (list): A list of tokens to be used for training the vocabulary.
        """
        # Apply case transformation based on the specified case
        if self._case == 'lower':
            tokens = map(str.lower, tokens)
        elif self._case == 'upper':
            tokens = map(str.upper, tokens)
        # 'both' case doesn't require transformation

        # Update the vocabulary mapping for tokens not already in _word2idx
        # Add tokens to the vocabulary if they are not already present
        for token in tokens:
            if token not in self._word2idx:
                self._word2idx[ token ] = len(self._word2idx)

        # Rebuild reverse lookup dictionary
        self._idx2word = { v: k for k, v in self._word2idx.items() }

    def encode( self, word ):

        """
        Encodes a word into its corresponding index in the vocabulary.

        Parameters:
        word (str): The word to be encoded.

        Returns:
        int: The index of the word (case-transformed if applicable) if found in the vocabulary,
             else returns index of the UNK token.
        """
        if self._case == 'lower':
            word = word.lower()
        elif self._case == 'upper':
            word = word.upper()

        return self._word2idx[ word ] if self._target else self._word2idx.get(word, self._word2idx[ self.UNK ])

        # """
        # Encodes a word into its corresponding index in the vocabulary.
        #
        # Parameters:
        # word (str): The word to be encoded.
        #
        # Returns:
        # int: The index of the lowercased word if found in vocabulary, else returns index of the UNK token.
        # """
        #
        # return self._word2idx[ word.lower() ] if self._target else self._word2idx.get(word.lower(), self._word2idx[ '<unk>' ])

    def decode( self, idx ):
        """
        Decodes an index into its corresponding word in the vocabulary.

        Parameters:
        idx (int): The index to be decoded.

        Returns:
        str: Word corresponding to provided index if found, else returns the UNK token.
        """
        return self._idx2word[ idx ] if self._target else self._idx2word.get(idx, self.UNK)
