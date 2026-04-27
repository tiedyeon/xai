import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import model
import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_idx = 0
SOS_token = 0
EOS_token = 1
hidden_size = 256
batch_size = 32

    # 전체 학습 루프
    # Adam 옵티마이저, NLLLoss(negative Log Likelihood Loss) 사용
def train(train_dataloader, model, n_epochs, learning_rate=0.0003):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_idx) # ignore_index = 0 -> 패딩 토큰(0번)에 대한 손실을 무시. 패딩은 의미 없는 채움값이니까 틀려도 패널티를 주지 않음

    for epoch in range(1, n_epochs + 1):
        loss = 0
        for iter, batch in enumerate(train_dataloader):
            # Batch tensors: [B, SeqLen]
            input_tensor  = batch[0]
            input_mask    = batch[1]
            target_tensor = batch[2]
            loss += train_step(input_tensor, input_mask, target_tensor,
                               model, optimizer, criterion)
        print('Epoch {} Loss {}'.format(epoch, loss / iter)) # 매 에폭마다 평균 손실을 출력


    # 한 배치의 학습 과정 
def train_step(input_tensor, input_mask, target_tensor, model,
               optimizer, criterion):
    optimizer.zero_grad() # 1. 이전 배치의 그래디언트를 초기화
    decoder_outputs, decoder_hidden = model(input_tensor, input_mask, target_tensor) # 2. 모델에 입력을 너허 예측값을 얻음(이때 target_tensor도 전달 -> Teacher Forcing활성화)

    # Collapse [B, Seq] dimensions for NLL Loss
    # decoder oupputs[B, Seq, Vocab]을 [B*Seq, Vocab]으로, target_tensor[B, SEq] 을 [B*Seq]로 펴서 NLLLoss에 넣음. NLLLoss가 2D입력을 기대하기 때문
    loss = criterion(
        decoder_outputs.view(-1, decoder_outputs.size(-1)), # [B, Seq, OutVoc] -> [B*Seq, OutVoc]
        target_tensor.view(-1) # [B, Seq] -> [B*Seq]
    )
    
    loss.backward() # 역전파
    optimizer.step() # 가중치 업데이트
    return loss.item()

    # 숫자 시퀀스를 단어 리스트로 변환하는 유틸리티. [2,3,4] ->["i","am","happy"]. Lang 객체의 index2word 사전을 참조
def ids2words(lang, ids):
    return [lang.index2word[idx] for idx in ids]

    # 학습 완료 후 모델이 실제로 번역을 잘 하는지 확인하는 함수
def greedy_decode(model, dataloader, input_lang, output_lang):
        # 계산을 끔(추론이니까)
    with torch.no_grad():
        batch = next(iter(dataloader))
        input_tensor  = batch[0]
        input_mask    = batch[1]
        target_tensor = batch[2]

        decoder_outputs, decoder_hidden = model(input_tensor, input_mask) # target_tensor 없이 입력만 넣음(추론이니까) -> target이 없으면 Teacher Forcing 이 꺼지고, 모델이 스스로 예측한 단어를 다음 입력으로 사용하는 greedy docoding 방식으로 동작
        topv, topi = decoder_outputs.topk(1) # topk(1) 로 매 스텝 가장 확률 높은 단어 선택
        decoded_ids = topi.squeeze()

        # 배치의 각 샘플에 대해 입력/정답/예측을 출력해서 비교
        for idx in range(input_tensor.size(0)):
            input_sent = ids2words(input_lang, input_tensor[idx].cpu().numpy())
            output_sent = ids2words(output_lang, decoded_ids[idx].cpu().numpy())
            target_sent = ids2words(output_lang, target_tensor[idx].cpu().numpy())
            print('Input:  {}'.format(input_sent))
            print('Target: {}'.format(target_sent))
            print('Output: {}'.format(output_sent))


if __name__ == '__main__':
    input_lang, output_lang, train_dataloader = load_data.get_dataloader(batch_size) # get_dataloader : 프랑스어 ->영어 데이터를 준비
    model = model.EncoderDecoder(hidden_size, input_lang.n_words, output_lang.n_words).to(device) # EncoderDecoder 모델 생성(hidden_size - 256)
    train(train_dataloader, model, n_epochs=20) # 20에폭 학습
    greedy_decode(model, train_dataloader, input_lang, output_lang) # 번역 결과 확인


