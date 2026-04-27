from io import open
import unicodedata
import string
import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


# 하나의 언어에 대한 단어 사전을 관리하는 클래스, Ex. 프랑스어용 Lang, 영어용 Lang
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}    # 단어를 숫자로 변환
        self.word2count = {}    # 각 단어가 데이터에서 몇 번 나왔는지 세는 용도
        self.index2word = { SOS_token: "SOS", EOS_token: "EOS"} # 숫자를 단어로 변환
        self.n_words = 2  # Count SOS and EOS

    # 문장을 공백 기준으로 쪼개서 각 단어를 addWord에 전달
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    # 처음 보는 단어면 새 인덱스를 부여하고 사전에 등록, 이미 있는 단어면 카운트만 1 올림
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # 데이터를 필터링함. 조건1:첫재 입력과 출력 문장 모두 10단어(MAX_LENGTH) 미만, 조건 2:영어 문장이 특정 접두사(eng_prefixes) 로 시작해야 됨 
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
    # 텍스트 전처리 함수, 프랑스어의 악센트 문자를 일반 ASCII로 변환
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
    # 텍스트 전처리 함수, 소문자로 바꾸고, 구두점 앞에 공백을 넣고, 알파벳과 구두점 외의 문자를 제거
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.

    # data/eng-fra.txt 파일을 읽어서 문장 쌍을 만듬, reverse=True로 호출하면 프랑스어->영어 순서로 뒤집음
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs




######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
    # readLangs -> filterPairs -> addSentence 순서대로 호출, 데이터 읽기, 필터링, 사전 구축을 한번에 수행, 최종적으로 입력 언어 사전, 출력 언어 사전, 문장 쌍 리스트 반환
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


# indexesFromSentence, tensorFromSentence, tensorsFromPair
# 문장을 숫자 시퀀스로 변환하는 함수들, input->Lang 사전 조회-> 인덱싱 -> EOS 추가 -> PyTorch 텐서로 변환, tensorsFromPair 는 입력-출력 쌍 한꺼번에 변환

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


    # 전체 데이터 파이프라인을 하나로 묶는 핵심 함수
def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)    # prepareData : 프랑스어 -> 영어 문장 쌍을 준비
    
    # 모든 문장을 고정길이(MAX_LENGTH)의 numpy 배열로 변환. 문장이 10단어보다 짧으면 나머지는 0(패딩)으로 채움
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)   
    input_mask = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_mask = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    # input_mask 를 만드는데, 실제 단어가 있는 위치는 1, 패딩인 위치는 0. 예를 들어 "je suis"(3단어) -> mask=[1,1,1,0,0,0,0,0,0,0]. 이 마스크는 나중에 Attention에서 패딩 위치에 주의를 기울이지 않도록 하는데 사용됨
    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        input_ids[idx, :len(inp_ids)] = inp_ids
        input_mask[idx, :len(inp_ids)] = 1
        target_ids[idx, :len(tgt_ids)] = tgt_ids
        target_mask[idx, :len(tgt_ids)] = 1
        
    # TensorDataset과 DataLoader로 감싸서 배치 단위로 뽑을 수 있게 반환
    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(input_mask).to(device),
                               torch.LongTensor(target_ids).to(device),
                               torch.LongTensor(target_mask).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader
