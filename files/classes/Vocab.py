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

        # Define special tokens similar to BERT
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]' 
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        
        # Define token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3

        # Dictionary mapping words to indices, blank if target is True else made with special tokens.
        self._word2idx = {} if self._target else {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.cls_token: self.cls_token_id,
            self.sep_token: self.sep_token_id
        }
        

        # Dictionary mapping indices to words, reverses lookup for the vocab.
        self._idx2word = { v: k for k, v in self._word2idx.items() }


    def train( self, tokens ):
        """
        Trains the vocabulary on a list of tokens.

        Parameters:
        tokens (list): A list of tokens to be used for training the vocabulary.
        """
        processed_tokens = []
        for token in tokens:
            # Skip case conversion for special tokens
            if token in [self.cls_token, self.sep_token, self.pad_token, self.unk_token]:
                processed_tokens.append(token)
            else:
                # Apply case transformation based on the specified case
                if self._case == 'lower':
                    processed_tokens.append(token.lower())
                elif self._case == 'upper':
                    processed_tokens.append(token.upper())
                else:  # 'both' case
                    processed_tokens.append(token)

        # Add tokens to the vocabulary if they are not already present
        for token in processed_tokens:
            if token not in self._word2idx:
                self._word2idx[token] = len(self._word2idx)

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
        if word in [self.cls_token, self.sep_token, self.pad_token, self.unk_token]:
            return self._word2idx[word]
        
        if self._case == 'lower':
            word = word.lower()
        elif self._case == 'upper':
            word = word.upper()

        return self._word2idx[ word ] if self._target else self._word2idx.get(word, self._word2idx[ self.unk_token ])

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
        return self._idx2word[ idx ] if self._target else self._idx2word.get(idx, self.unk_token)
