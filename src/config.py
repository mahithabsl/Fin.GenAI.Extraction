config = {

    'embeddings': {
        'sentence_transformer': {
            'model': 'all-mpnet-base-v2'
        }
    },
    'chunking': {
        'method': 'nltk',  # Options: 'gpt2', 'nltk', 'character_and_token'
        'model': 'gpt2',
        'chunk_size': 1000,
        'chunk_overlap': 50,
        'tokens_per_chunk': 200,
        'nltk': {
            'w': 20,  # TextTiling window size
            'k': 10   # TextTiling smoothing parameter
        }
    }
}