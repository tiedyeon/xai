import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 인코더; 입력 문장(프랑스어)을 읽어서 의미를 압축
class EncoderRNN(nn.Module):
        # nn.Embedding 으로 단어 인덱스를 hidden_size 차원의 벡터로 변환하는 임베딩 층을 만들고, nn.GRU로 순환 신경망 층을 만듬
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # 입력 단어 인덱스 -> 임베딩 벡터로 변환 -> GRU 통화
        # GRU 가 반환하는 output은 모든 시점의 은닉 상태[B,Seq,D]이고, hidden은 마지막 시점의 [1,B,D]
        # [B,seq,D] -> Batch size : 한 번에 처리하는 문장 수, 시퀀스 길이(Seq) 문장의 단어 수, D는 임베딩 차원 : [1,3,4] -> 1개의 문장의 3개의 단어를 4차원의 임베딩으로 나타냄
        # [1,B,D] -> 1 : GRU의 층 수, 배치 크기(문장 개수), 은닉 상태의 차원 : output의 마지막 행 값
        # output은 나중에 Attention에서 사용되고, hidden은 디코더의 초기 상태로 전달
        # EX. "je suis etudiant"(3단어) -> 임베딩 -> GRU -> output에는 각 단어 위치별 은닉 상태 3개가, hidden에는 마지막("etudiant") 시점의 은닉 상태 1개가 담김
    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        return output, hidden


    # Attention 없는 기본 디코더, 여기 코드에서는 사용하지 않음. 비교용으로 남겨둔것
class DecoderRNN(nn.Module):
    # Standard non-attentional decoder
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

        # SOS 토큰부터 시작해 max_len만큼 반복하며 한 단어씩 생성
    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, SOS_token=0, max_len=10):
        # Teacher forcing if given a target_tensor, otherwise greedy.
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden # TODO: Consider bridge
        decoder_outputs = []

        for i in range(max_len):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            # Teacher Forcing : target_tensor 가 주어지면(학습 시) 이전 스텝의 정답 단어를 다읍 입력으로 사용, 주어지지 않으면(추론 시) 모델이 예측한 단어를 다음 입력으로 사용
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(-1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1) # [B, Seq, OutVocab]
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden

        # 한 타임스텝 처리 : 입력 -> 임베딩 -> ReLU -> GRU -> 선형 층 -> 출력, Attention 없이 GRU의 은닉 상태만으로 다음 단어 예측
    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

    # Bahdanau(Addictiva) Attention 메커니즘
class BahdanauAttention(nn.Module):
        # 세 개의 선형 층 정의
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size) # 디코더 쿼리(query)변환
        self.W2 = nn.Linear(hidden_size, hidden_size) # 인코더 출력(values) 변환
        self.V = nn.Linear(hidden_size, 1) # 최종 스코어를 스칼라로 만듬
        self.W3 = nn.Linear(hidden_size, 1)

        # query[B,1,D] : 디코더의 현재 은닉 상태. "지금 내가 어떤 단어를 생성하려는 상태"
        # values[B,M,D] : 인코더의 모든 시점 출력. "입력 문장의 각 단어별 정보"
        # mask[B,M] : 패딩 마스크, 실제 단어면 1, 패딩이면 0
    def forward(self, query, values, mask):
        # Additive attention
        # 1단계 - 스코어 계산
        # query 와 values를 각각 선형 변환한뒤 더하고, tanh를 거쳐 V로 스칼라를 뽑음
        # "현재 디코더 상태에서 입력 문장의 각 단어가 얼마나 관련 있는지"의 점수
        scores = self.V(torch.tanh(self.W1(query) + self.W2(values)))
        scores = scores.squeeze(2).unsqueeze(1) # [B, M, 1] -> [B, 1, M]

        # 아래의 Dot-Product, Cosine Similarity 는  대안적 방안, 여기서는  Addictive attention 활용
        # Dot-Product Attention: score(s_t, h_i) = s_t^T h_i
        # Query [B, 1, D] * Values [B, D, M] -> Scores [B, 1, M]
        # scores = torch.bmm(query, values.permute(0,2,1))

        # Cosine Similarity: score(s_t, h_i) = cosine_similarity(s_t, h_i)
        # scores = F.cosine_similarity(query, values, dim=2).unsqueeze(1)

        # Mask out invalid positions.
        # 2단계 - 마스킹
        scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf')) # 패딩 위치의 스코어를 -무한대로 설정. 다음 단계의 softmax에서 해당 위치의 가중치가 0이 됨, 패딩에는 절대 주의를 기울이지 않게 됨

        # 3단계 - Attention 가중치(alphas)
        # Attention weights
        alphas = F.softmax(scores, dim=-1) # 스코어를 softmax로 정규화, 합이 1인 확률 분포로 만듬. 예를 들어 "je"에 0.1, "suis"에 0.7, "etudiant"에 0.2 -> "suis"에 주목하겠다는 뜻
        
        # 4단계 - Context 벡터
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, values) # 가중치를 values에 곱해서 가중합을 구함. 결과는 입력 문장 전체의 정보가 현재 디코더 상태에 맞게 요약된 하나의 벡터

        # context shape: [B, 1, D], alphas shape: [B, 1, M]
        return context, alphas


    # Attention이 적용된 디코더. DecoderRNN과 구조가 비슷하지만, 매 스탭마다 BahdanuaAttention을 호출하는 것이 핵심 차이
class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True) # 이 점에 주목. 임베딩 벡터(hidden_size)와 context 벡터(hidden_size)를 이어붙이기(concatenate) 때문
        self.out = nn.Linear(hidden_size, output_size)


        # DecoderRNN의 forward와 거의 동일한 루프이나, forward_step에 encoder_outputs와 input_mask를 추가로 전달
    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, SOS_token=0, max_len=10):
        # Teacher forcing if given a target_tensor, otherwise greedy.
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden # TODO: Consider bridge
        decoder_outputs = []

        for i in range(max_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, input_mask)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(-1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1) # [B, Seq, OutVocab]
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden

        # 핵심 차이
        # 1. 디코더 은닉 상태를 query로 변환 : hiddden[1,B,D] -> query[B,1,D]
        # 2. self.attention(query, encoder_outputs, input_mask) 로 context 벡터를 구함
        # 3. 현재 입력 단어의 임베딩과 context 벡터를 이어붙여서 [B,1,2*D] 로 만듬
        # 4. 이걸 GRU에 통과시킨 후 선형 층으로 다음 단어를 예측
        # DecoderRNN은 "인코더 마지막 은닉 상태만 보고" 번역하지만, AttnDecoder는 "매 단어 생성 시마다 입력 문장의 어디를 봐야 할지 동적으로 결정". 이것이 Attntion의 핵심
    def forward_step(self, input, hidden, encoder_outputs, input_mask):
        # encoder_outputs: [B, Seq, D]
        query = hidden.permute(1, 0, 2) # [1, B, D] --> [B, 1, D]
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)
        embedded = self.embedding(input)
        attn = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(attn, hidden)
        output = self.out(output)
        # output: [B, 1, OutVocab]
        return output, hidden, attn_weights


    # 인코더와 디코더를 하나로 묶는 래퍼 클래스
    # forward 에서 입력을 인코더에 통과시키고, 그 출력을 디코더에 전달하는 흐름만 정의함
class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderRNN(input_vocab_size, hidden_size)
        self.decoder = AttnDecoder(hidden_size, output_vocab_size)
        # self.decoder = DecoderRNN(hidden_size, output_vocab_size) # DecoderRNN은 미사용

    def forward(self, inputs, input_mask, targets=None):
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        decoder_outputs, decoder_hidden = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, targets)
        return decoder_outputs, decoder_hidden
