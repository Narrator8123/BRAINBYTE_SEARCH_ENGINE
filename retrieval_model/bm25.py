import random
import pandas as pd
import json
from scipy.sparse import load_npz
import numpy as np
from scipy.sparse import csr_matrix


def compute_idf(matrix, document_count):
    binarized_matrix = matrix.copy()
    binarized_matrix.data = np.ones_like(binarized_matrix.data)
    doc_freq = np.array(binarized_matrix.sum(axis=0)).flatten()
    idf = np.log((document_count + 1) / (doc_freq + 1)) + 1
    return csr_matrix(idf)


def bm25_score(matrix, k1=1.5, b=0.75):
    doc_len = np.array(matrix.sum(axis=1)).flatten()
    avg_doc_len = np.mean(doc_len)
    tf = matrix.multiply(csr_matrix(k1 / (1 - b + b * (doc_len / avg_doc_len))[:, None]))
    doc_count = matrix.shape[0]
    idf = compute_idf(matrix, doc_count)
    bm25 = tf.multiply(idf)
    return bm25


def get_document(document_id):
    (string1, string2, bundle_num, position) = document_id.split('_')
    assert string1 == 'msmarco' and string2 == 'doc'

    with open(f'../data/msmarco_v2_doc/msmarco_doc_{bundle_num}', 'rt', encoding='utf8') as in_fh:
        in_fh.seek(int(position))
        json_string = in_fh.readline()
        document = json.loads(json_string)
        assert document['docid'] == document_id
        return document


def sample_get_document(docid):
    with open('../data/sample_data/sample_doc_byte_index.json', 'r', encoding='utf-8') as file:
        byte_index = json.load(file)
    position = byte_index.get(str(docid))
    if position is not None:
        with open('../data/sample_data/sample_doc', 'r', encoding='utf-8') as file:
            file.seek(position)
            line = file.readline()
            return json.loads(line)
    return None


def retrieve_function(query, precision=0.1):
    numbers = list(range(60))
    id = random.sample(numbers, int(precision * 60))
    total_score = np.array([])
    total_ids = []
    for i in id:
        corpus_id = f"{i:02}"
        documents_vectorized = load_npz(f"../data/msmarco_v2_doc_vectorized/msmarco_doc_{corpus_id}.npz")
        vocabulary_df = pd.read_csv(f"../data/msmarco_v2_doc_vectorized/msmarco_doc_{corpus_id}_vocabulary.csv")
        vocabulary = vocabulary_df['vocabulary'].tolist()
        ids_df = pd.read_csv(f"../data/msmarco_v2_doc_vectorized/msmarco_doc_{corpus_id}_ids.csv")
        ids = ids_df['document_id'].tolist()
        q_terms = query.lower().split(' ')
        q_index = [vocabulary.index(term) for term in q_terms if term in vocabulary]
        bm25_scores = bm25_score(documents_vectorized)
        filtered_scores = bm25_scores[:, q_index]
        doc_scores_sum = np.array(filtered_scores.sum(axis=1)).flatten()
        total_score = np.hstack((total_score, doc_scores_sum))
        total_ids.extend([corpus_id + str(id) for id in ids])
    top_doc_indices = np.argsort(-total_score)[:1000]
    top_doc_ids = [total_ids[i] for i in top_doc_indices]
    top_doc = []
    for i, doc_id in enumerate(top_doc_ids):
        document = get_document(f'msmarco_doc_{doc_id[:2]}_{doc_id[2:]}')
        top_doc.append({document.get('docid'): [document.get('title'), document.get('url'), f'BM25_rank{i + 1}',
                                                document.get('body')]})
    return top_doc


def sample_retrieve_function(query):
    documents_vectorized = load_npz(f"../data/sample_data/sample_doc_vectorized/sample_doc.npz")
    vocabulary_df = pd.read_csv(f"../data/sample_data/sample_doc_vectorized/sample_doc_vocabulary.csv")
    vocabulary = vocabulary_df['vocabulary'].tolist()
    ids_df = pd.read_csv(f"../data/sample_data/sample_doc_vectorized/sample_doc_ids.csv")
    ids = ids_df['document_id'].tolist()
    q_terms = query.lower().split(' ')
    q_index = [vocabulary.index(term) for term in q_terms if term in vocabulary]
    bm25_scores = bm25_score(documents_vectorized)
    filtered_scores = bm25_scores[:, q_index]
    doc_scores_sum = np.array(filtered_scores.sum(axis=1)).flatten()
    top_doc_indices = np.argsort(-doc_scores_sum)[:300]
    top_doc_ids = [ids[i] for i in top_doc_indices]
    top_doc = []
    for i, doc_id in enumerate(top_doc_ids):
        document = sample_get_document(doc_id)
        top_doc.append({document.get('docid'): [document.get('title'), document.get('url'), f'rank{i + 1}',
                                                document.get('body')[:512]]})
    return top_doc

