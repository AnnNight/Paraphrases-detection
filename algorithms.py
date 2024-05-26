import pickle
import sql_request
from langchain_chroma import Chroma
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torch
import os
import unstructured
import pandas as pd
import chromadb

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def find_user(login, password) -> int:
    result = sql_request.get_user(login, password)
    if result >= 0:
        return result
    else:
        return -1


def find_login(login: str) -> int:
    result = sql_request.find_login(login)
    if result >= 0:
        return result
    else:
        return -1


def create_user(login: str, password, phoneNumber, email) -> int:
    result = sql_request.create_user(login, password, phoneNumber, email)
    if result >= 0:
        return result
    else:
        return -1


def compare_texts(text1, text2, model, tokenizer):
    batch = tokenizer(text1, text2, return_tensors='pt').to(model.device)
    with torch.no_grad():
        proba = torch.softmax(model(**batch).logits, -1).cpu().numpy()
    return proba[0]  # p(non-paraphrase), p(paraphrase)


class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.roberta = AutoModel.from_pretrained('blinoff/roberta-base-russian-v0')
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.fc(features)
        return output


class AlbertClass(torch.nn.Module):
    def __init__(self):
        super(AlbertClass, self).__init__()
        self.roberta = AutoModel.from_pretrained('linhd-postdata/alberti-bert-base-multilingual-cased')
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.fc(features)
        return output


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = AutoModel.from_pretrained('ai-forever/ruBert-base')
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.fc(features)
        return output


def check_paraphrased_text(type_of_check, user_id, filesToBase, check_id=0):
    global device
    model_name = "ai-forever/sbert_large_nlu_ru"
    model_kwargs = {'device': device}
    embedding_function = SentenceTransformerEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )

    db_directory = './data/common_database'
    if not os.path.exists(db_directory):
        os.makedirs(db_directory)
    chroma_client = chromadb.PersistentClient(path=db_directory)
    collection_name = "my_collection"
    common_db = chroma_client.get_or_create_collection(collection_name)
    common_db = Chroma(collection_name, embedding_function)
    if type_of_check == "Проверка всех загруженных файлов относительно всех":
        pathdir = "./data/" + str(user_id) + "/" + str(check_id)
        if not os.path.exists(pathdir + "/src"):
            os.makedirs(pathdir + "/src")
        loader = DirectoryLoader(pathdir + "/src")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, separators=['.', '?', '!'])
        docs = text_splitter.split_documents(documents)
        model_name = "ai-forever/sbert_large_nlu_ru"
        model_kwargs = {'device': device}
        embedding_function = SentenceTransformerEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )

        if filesToBase:
            for doc in docs:
                if common_db:
                    common_db.add_documents([doc])
                else:
                    common_db = Chroma.from_documents([doc], embedding_function, persist_directory=db_directory)

        outputs = []
        db = Chroma.from_documents(docs, embedding_function, persist_directory=pathdir + '/Chroma_db')
        for doc in docs:
            outputs.append(db.similarity_search(doc.page_content))

        model_name = 's-nlp/ruRoberta-large-paraphrase-v1'
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True)

        for i in range(len(docs)):
            outputs[i] = list(filter(lambda x: docs[i].metadata["source"] != x.metadata["source"], outputs[i]))
        final_results = []

        for i in range(len(docs)):
            best = (0, None)
            for output in outputs[i]:
                r = compare_texts(str(docs[i].page_content), str(output.page_content), model, tokenizer)[1]
                if r > 0.5 and r > best[0]:
                    best = (r, output)
            final_results.append([docs[i], best[1]])

        last_final_results = {}
        files = os.listdir(pathdir + "/src")
        for file in files:
            last_final_results[file] = []
        for result in final_results:
            name = result[0].metadata['source'].split("\\")[-1]
            if result[1] is not None:
                name2 = result[1].metadata['source'].split("\\")[-1]
            else:
                name2 = None
            last_final_results[name].append((result[0].page_content, name2))
        return last_final_results, "Проверка завершена, перейдите на следующую вкладку, чтобы узнать результат"

    elif type_of_check == "Проверка одного файла относительно загруженных":  # Один со всеми
        pathdir = "./data/" + str(user_id) + "/" + str(check_id)
        if not os.path.exists(pathdir + "/src"):
            os.makedirs(pathdir + "/src")
        if not os.path.exists(pathdir + "/susp"):
            os.makedirs(pathdir + "/susp")
        loader = DirectoryLoader(pathdir + "/src")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0, separator='.')
        docs = text_splitter.split_documents(documents)
        model_name = "ai-forever/sbert_large_nlu_ru"
        model_kwargs = {'device': device}
        embedding_function = SentenceTransformerEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )

        if filesToBase:
            for doc in docs:
                if common_db:
                    common_db.add_documents([doc])
                else:
                    common_db = Chroma.from_documents([doc], embedding_function, persist_directory=db_directory)
        db = Chroma.from_documents(docs, embedding_function, persist_directory=pathdir + '/Croma_db')
        loader = DirectoryLoader(pathdir + "/susp")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0, separator='.')
        inputs = text_splitter.split_documents(documents)

        outputs = []
        for each in inputs:
            outputs.append(db.similarity_search(each.page_content))

        model_name_p = 's-nlp/ruRoberta-large-paraphrase-v1'
        model_p = AutoModelForSequenceClassification.from_pretrained(model_name_p)
        tokenizer_p = AutoTokenizer.from_pretrained(model_name_p, padding='max_length', truncation=True,
                                                    model_max_length=512)

        final_results = []
        for i in range(len(inputs)):
            best = (0, None)
            for output in outputs[i]:
                r = compare_texts(str(inputs[i].page_content), str(output.page_content), model_p, tokenizer_p)[1]
                if r > 0.5 and r > best[0]:
                    best = (r, output)
            final_results.append([inputs[i], best[1]])
        last_final_results = {}
        files = os.listdir(pathdir + "/susp")
        for file in files:
            last_final_results[file] = []
        for result in final_results:
            name = result[0].metadata['source'].split("\\")[-1]
            if result[1] is not None:
                name2 = result[1].metadata['source'].split("\\")[-1]
            else:
                name2 = None
            last_final_results[name].append((result[0].page_content, name2))
        return last_final_results, "Проверка завершена, перейдите на следующую вкладку, чтобы узнать результат"
    elif type_of_check == "Проверка файла в общей базе":
        model_name = "ai-forever/sbert_large_nlu_ru"
        model_kwargs = {'device': device}
        embedding_function = SentenceTransformerEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )
        pathdir = "./data/" + str(user_id) + "/" + str(check_id)
        if not os.path.exists(pathdir + "/src"):
            os.makedirs(pathdir + "/src")
        if not os.path.exists(pathdir + "/susp"):
            os.makedirs(pathdir + "/susp")
        loader = DirectoryLoader(pathdir + "/susp")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0, separator='.')
        docs = text_splitter.split_documents(documents)
        outputs = []
        for doc in docs:
            outputs.append(common_db.similarity_search(doc.page_content))

        model_name = 's-nlp/ruRoberta-large-paraphrase-v1'
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True)

        for i in range(len(docs)):
            outputs[i] = list(filter(lambda x: docs[i].metadata["source"] != x.metadata["source"], outputs[i]))
        final_results = []

        for i in range(len(docs)):
            best = (0, None)
            for output in outputs[i]:
                r = compare_texts(str(docs[i].page_content), str(output.page_content), model, tokenizer)[1]
                if r > 0.5 and r > best[0]:
                    best = (r, output)
            final_results.append([docs[i], best[1]])

        last_final_results = {}
        files = os.listdir(pathdir + "/susp")
        for file in files:
            last_final_results[file] = []
        for result in final_results:
            name = result[0].metadata['source'].split("\\")[-1]
            if result[1] is not None:
                name2 = result[1].metadata['source'].split("\\")[-1]
            else:
                name2 = None
            last_final_results[name].append((result[0].page_content, name2))
        return last_final_results, "Проверка завершена, перейдите на следующую вкладку, чтобы узнать результат"


def process_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, return_token_type_ids=True)
    outputs = model(**inputs)
    return outputs


def preprocess_text(text, tokenizer):
    return tokenizer(text, return_tensors='pt', return_token_type_ids=True)


def get_model_outputs(input_ids, attention_mask, model, token_type_ids):
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
    return outputs


def check_generated_text(type_of_check, filesToBase, user_id='0', check_id='0'):
    global device
    pathdir = "./data/" + str(user_id) + "/" + str(check_id)
    if type_of_check == "Проверка одного файла":
        if not os.path.exists(pathdir + "/susp"):
            os.makedirs(pathdir + "/susp")
        loader = DirectoryLoader(pathdir + "/susp")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0, separator='.')
        docs = text_splitter.split_documents(documents)
        tokenizer_R = AutoTokenizer.from_pretrained('blinoff/roberta-base-russian-v0', max_len=512)
        model_R = RobertaClass()
        model_R.load_state_dict(torch.load('models/Model_R.pth', map_location=torch.device(device)))

        tokenizer_A = AutoTokenizer.from_pretrained(
            'linhd-postdata/alberti-bert-base-multilingual-cased', max_len=512)
        model_A = AlbertClass()
        model_A.load_state_dict(torch.load('models/Model_A.pth', map_location=torch.device(device)))

        tokenizer_B = AutoTokenizer.from_pretrained('ai-forever/ruBert-base', max_len=512)
        model_B = BERTClass()
        model_B.load_state_dict(torch.load('models/Model_B.pth', map_location=torch.device(device)))
        with open('models/classifierMG.pickle', 'rb') as f:
            classifier = pickle.load(f)
        test_dataframe = pd.DataFrame()
        test_dataframe['id'] = 0
        test_dataframe['R_0'] = 0
        test_dataframe['R_1'] = 0
        test_dataframe['A_0'] = 0
        test_dataframe['A_1'] = 0
        test_dataframe['B_0'] = 0
        test_dataframe['B_1'] = 0
        i = 0
        for doc in docs:
            # test_dataframe.loc[i] = [i, doc.page_content]
            R_input = preprocess_text(doc.page_content, tokenizer_R)
            R_input_ids = R_input['input_ids'].to(device)
            R_attention_mask = R_input['attention_mask'].to(device)
            R_token_type_ids = R_input['token_type_ids'].to(device)
            R_outputs = get_model_outputs(R_input_ids, R_attention_mask, model_R, R_token_type_ids)
            R_fin_outputs = torch.sigmoid(R_outputs).cpu().detach().numpy().tolist()
            A_input = preprocess_text(doc.page_content, tokenizer_A)
            A_input_ids = A_input['input_ids'].to(device)
            A_attention_mask = A_input['attention_mask'].to(device)
            A_token_type_ids = A_input['token_type_ids'].to(device)
            A_outputs = get_model_outputs(A_input_ids, A_attention_mask, model_A, A_token_type_ids)
            A_fin_outputs = torch.sigmoid(A_outputs).cpu().detach().numpy().tolist()
            B_input = preprocess_text(doc.page_content, tokenizer_B)
            B_input_ids = B_input['input_ids'].to(device)
            B_attention_mask = B_input['attention_mask'].to(device)
            B_token_type_ids = B_input['token_type_ids'].to(device)
            B_outputs = get_model_outputs(B_input_ids, B_attention_mask, model_B, B_token_type_ids)
            B_fin_outputs = torch.sigmoid(B_outputs).cpu().detach().numpy().tolist()
            test_dataframe.loc[i] = [i, R_fin_outputs[0][0], R_fin_outputs[0][1], A_fin_outputs[0][0],
                                     A_fin_outputs[0][1],
                                     B_fin_outputs[0][0], B_fin_outputs[0][1]]

            i = i + 1
        test_ids = test_dataframe['id']
        test_dataframe = test_dataframe.drop(['id'], axis=1)
        test_preds = classifier.predict(test_dataframe)

        last_final_results = {}
        files = os.listdir(pathdir + "/susp")
        for file in files:
            last_final_results[file] = []
        for i in range(len(docs)):
            name = files[0]
            if test_preds[i] > 0:
                pred = 'M'
            else:
                pred = None
            last_final_results[name].append((docs[i].page_content, pred))
        return last_final_results, "Проверка завершена, перейдите на следующую вкладку, чтобы узнать результат"

    elif type_of_check == "Проверка нескольких файлов":
        if not os.path.exists(pathdir + "/src"):
            os.makedirs(pathdir + "/src")
        loader = DirectoryLoader(pathdir + "/src")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0, separator='.')
        docs = text_splitter.split_documents(documents)
        tokenizer_R = AutoTokenizer.from_pretrained('blinoff/roberta-base-russian-v0', max_len=512)
        model_R = RobertaClass()
        model_R.load_state_dict(torch.load('models/Model_R.pth', map_location=torch.device(device)))

        tokenizer_A = AutoTokenizer.from_pretrained(
            'linhd-postdata/alberti-bert-base-multilingual-cased', max_len=512)
        model_A = AlbertClass()
        model_A.load_state_dict(torch.load('models/Model_A.pth', map_location=torch.device(device)))

        tokenizer_B = AutoTokenizer.from_pretrained('ai-forever/ruBert-base', max_len=512)
        model_B = BERTClass()
        model_B.load_state_dict(torch.load('models/Model_B.pth', map_location=torch.device(device)))
        with open('models/classifierMG.pickle', 'rb') as f:
            classifier = pickle.load(f)
        result = []
        test_dataframe = pd.DataFrame()
        test_dataframe['id'] = 0
        # test_dataframe['content'] = ""
        test_dataframe['R_0'] = 0
        test_dataframe['R_1'] = 0
        test_dataframe['A_0'] = 0
        test_dataframe['A_1'] = 0
        test_dataframe['B_0'] = 0
        test_dataframe['B_1'] = 0
        i = 0
        for doc in docs:
            # test_dataframe.loc[i] = [i, doc.page_content]
            R_input = preprocess_text(doc.page_content, tokenizer_R)
            R_input_ids = R_input['input_ids'].to(device)
            R_attention_mask = R_input['attention_mask'].to(device)
            R_token_type_ids = R_input['token_type_ids'].to(device)
            R_outputs = get_model_outputs(R_input_ids, R_attention_mask, model_R, R_token_type_ids)
            R_fin_outputs = torch.sigmoid(R_outputs).cpu().detach().numpy().tolist()
            A_input = preprocess_text(doc.page_content, tokenizer_A)
            A_input_ids = A_input['input_ids'].to(device)
            A_attention_mask = A_input['attention_mask'].to(device)
            A_token_type_ids = A_input['token_type_ids'].to(device)
            A_outputs = get_model_outputs(A_input_ids, A_attention_mask, model_A, A_token_type_ids)
            A_fin_outputs = torch.sigmoid(A_outputs).cpu().detach().numpy().tolist()
            B_input = preprocess_text(doc.page_content, tokenizer_B)
            B_input_ids = B_input['input_ids'].to(device)
            B_attention_mask = B_input['attention_mask'].to(device)
            B_token_type_ids = B_input['token_type_ids'].to(device)
            B_outputs = get_model_outputs(B_input_ids, B_attention_mask, model_B, B_token_type_ids)
            B_fin_outputs = torch.sigmoid(B_outputs).cpu().detach().numpy().tolist()
            test_dataframe.loc[i] = [i, R_fin_outputs[0][0], R_fin_outputs[0][1], A_fin_outputs[0][0],
                                     A_fin_outputs[0][1],
                                     B_fin_outputs[0][0], B_fin_outputs[0][1]]

            i = i + 1
        test_ids = test_dataframe['id']
        test_dataframe = test_dataframe.drop(['id'], axis=1)
        test_preds = classifier.predict(test_dataframe)

        last_final_results = {}
        files = os.listdir(pathdir + "/src")
        for file in files:
            last_final_results[file] = []
        for i in range(len(docs)):
            name = docs[i].metadata['source'].split("\\")[-1]
            if test_preds[i] > 0:
                pred = 'M'
            else:
                pred = None
            last_final_results[name].append((docs[i].page_content, pred))
        return last_final_results, "Проверка завершена, перейдите на следующую вкладку, чтобы узнать результат"
