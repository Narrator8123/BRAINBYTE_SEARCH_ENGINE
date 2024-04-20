from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import pandas as pd
from retrieval_model.bm25 import retrieve_function, sample_retrieve_function, sample_get_document
from retrieval_model.evaluation import calculate_n_dcg
import statistics
import string
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class RankingModel(nn.Module):
    def __init__(self):
        super(RankingModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.score_classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        score = self.score_classifier(pooled_output)
        return score


tokenizer = BertTokenizer.from_pretrained("../data/bert")

model = RankingModel()
model.load_state_dict(torch.load("../data/bert/bert_weights.pth", map_location=torch.device('cpu')))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def find_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    synonyms.add(word)
    return list(synonyms)


stop_words = set(stopwords.words('english'))


def retrieve_function(query, test=True):
    query_no_punctuation = query.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(query_no_punctuation)
    filtered_query = [word for word in words if not word.lower() in stop_words]
    keywords_synonyms = []
    for word in filtered_query:
        syn = find_synonyms(word)
        keywords_synonyms.extend(syn)
    extended_query = ' '.join(keywords_synonyms)

    if not test:
        documents = retrieve_function(query, precision=0.1)
    else:
        documents = sample_retrieve_function(extended_query)

    max_length = 512
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []

    for document in documents:
        (_, value), = document.items()
        inputs = tokenizer(
            query + " [SEP] " + value[3],
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids_list.append(inputs['input_ids'].to(device))
        attention_mask_list.append(inputs['attention_mask'].to(device))
        token_type_ids_list.append(inputs['token_type_ids'].to(device))

    scores = []
    with torch.no_grad():
        for i in range(len(documents)):
            predictions = model(input_ids_list[i], attention_mask_list[i], token_type_ids_list[i])
            scores.append(predictions.squeeze().item())

    doc_scores = list(zip(documents, scores))
    doc_scores_sorted = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:100]
    t100_doc, _ = zip(*doc_scores_sorted)
    t100_keys = [list(doc.keys())[0] for doc in t100_doc]
    id_scores = list(zip(t100_keys, scores))
    columns_name = ['qid', 'query']
    df = pd.read_csv('../data/sample_data/sample_queries.tsv', delimiter='\t', names=columns_name)
    qid = df.loc[df['query'] == query, 'qid'].iloc[0]
    ground_truth_scores = []
    predict_scores = []
    with open('../data/sample_data/sample_top100.txt', 'r', encoding='utf-8') as file:
        for line in file:
            t100 = line.split()
            if t100[0] == str(qid):
                ground_truth_scores.append(float(t100[4]))
                predict_score = float(next((score for doc_id, score in id_scores if doc_id == t100[2]), 0))
                predict_scores.append(predict_score)
    precision = len(predict_scores) - predict_scores.count(0.0)
    n_dcg = calculate_n_dcg(ground_truth_scores, predict_scores)
    print(f'Precision@100: {precision}')
    print(f'nDCG: {n_dcg:.2f}')

    doc_info = []
    bert_rank = 1
    for doc, score in doc_scores_sorted:
        (key, value), = doc.items()
        document = sample_get_document(key[15:])
        doc_info.append({document.get('docid'): [document.get('title'), document.get('url'),
                                                 document.get('body')[:150]]})
        print(f"Document: {key}, Score: {score}, BM25: {value[2]}, Bert: rank{bert_rank}")
        bert_rank += 1
    return doc_info, precision, n_dcg


'''
precisions = []
nDCG = []
with open('../sample_data/sample_queries.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        query = line.strip().split('\t')
        _, p, n = retrieve_function(query[1])
        precisions.append(p)
        nDCG.append(n)

print('-------------------------------')
print(statistics.mean(precisions))
print(statistics.mean(nDCG))
'''