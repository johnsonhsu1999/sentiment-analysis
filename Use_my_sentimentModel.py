from My_Sentiment_analysis import MLPmodel, preprocess,token_to_idx
import torch

#parameters
wordlist_len = 21402
device = torch.device("cpu")
embedding_dim = 32
hidden_dim = 64
num_class = 2
epoch_num = 50


#1. load model weight
model = MLPmodel(vocab_len=wordlist_len,embedding_dim=embedding_dim,hidden_dim=hidden_dim,num_class=num_class)
model.load_state_dict(torch.load('nn/models/my_model.pth'))

#2. load word_list and tokenize the input string 因為model input為tokenize的sentence所以要load wordlist及tokenize words
def tokenize(text):
    _,_,_,_,wordlist = preprocess()
    inputs = token_to_idx(wordlist,text,number_of_tokens=20)
    inputs = inputs.view(1,inputs.shape[0]) #dim must = (batch num, len of tokens)
    return inputs

#------------------------------------------------------------------------------------
#3. predict
test_data = [
    "I feel miserable and lonely in this gloomy weather.",                                      #1 neg
    "The food at that restaurant was terrible, and the service was even worse.",                #1 neg
    "I can't stand his rude and disrespectful behavior.",                                       #1 neg
    "The movie was a complete disappointment, with a predictable plot and terrible acting.",    #1 neg
    "I'm exhausted and frustrated after dealing with all these problems.",                      #1 neg
    "I'm overjoyed and excited about my upcoming vacation.",                                    #0 pos
    "The party was fantastic, with great music and delicious food.",                            #0 pos
    "I feel grateful and blessed to have such supportive friends and family.",                  #0 pos
    "The concert was amazing, with electrifying performances and an enthusiastic crowd.",       #0 pos
    "I'm thrilled and proud of my accomplishments in completing that difficult task."           #0 pos
]
real_answer = [1,1,1,1,1,0,0,0,0,0]
test_answer = []
for sent in test_data:
    res = model(tokenize(sent))
    predict_index = torch.argmax(res)
    test_answer.append(int(predict_index))
    ans = 'positive' if predict_index==0 else 'negative'
    print("prediction is : ",ans)

#acc
ans = 0
for i in range(10):
    print(real_answer[i], test_answer[i])
    if real_answer[i]==test_answer[i]:
        ans += 1
    print("ans = ",ans)
print("accuracy = ",ans*10,"%")