import nltk
import torch
from torch.utils.data import TensorDataset, DataLoader
from nltk.corpus import sentence_polarity
import numpy as np
from nltk.corpus import stopwords
from torch.nn import Module
import torch.nn.functional as F
import matplotlib.pyplot as plt

#load data, split data, tokenize data
def preprocess(): 
    #(1) make word_list, numerize the corpus and adjust the length of input data
    word_list = dict()
    #Stopword = set(stopwords.words('english'))
    index = 1
    word_list["<unk>"]=0
    data = sentence_polarity.sents()
    for review in data:
        for word in review:
            if word  not in word_list:
                word_list[word]=index
                index+=1

    
    #(2)split dataset into training and testing, tokenize them and padding
    def tokenize(datas, wordlist, max_len):
        res = []
        for review in datas:  #將這個batch裡面的每個review都給numerize
            if len(review)<max_len:
                review = [wordlist[word] for word in review] + list(np.zeros(max_len-len(review)))
                res.append(review)
            else:
                review = [wordlist[word] for word in review]
                res.append(review[:max_len])    
        return res
    


    max_len = 20   
    x_train = sentence_polarity.sents(categories="pos")[:4000] + sentence_polarity.sents(categories="neg")[:4000]
    y_train = [1 for i in range(4000)]+[0 for i in range(4000)]
    x_test = sentence_polarity.sents(categories="pos")[4000:] + sentence_polarity.sents(categories="neg")[4000:]
    y_test = [1 for i in range(1331)] + [0 for i in range(1331)]
    #for training, we use whole data to training 
    x_train = sentence_polarity.sents(categories="pos") + sentence_polarity.sents(categories="neg")
    y_train = [1 for i in range(5331)]+[0 for i in range(5331)]
    
    x_train = tokenize(datas=x_train, wordlist=word_list, max_len=max_len)
    x_test = tokenize(datas=x_test, wordlist=word_list, max_len=max_len)

    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)
    return x_train, y_train, x_test, y_test, word_list

#給外部去轉換wordlist index用的
def token_to_idx(word_list,sentences,number_of_tokens):
    res = []
    for sent in nltk.sent_tokenize(sentences):
        for word in nltk.word_tokenize(sent):
            if word in word_list:
                res.append(word_list[word])
            else:
                res.append(word_list["<unk>"])
    res += [0 for item in range(number_of_tokens-len(res))] #padding
    res = torch.tensor(res[:number_of_tokens])
    return res


class MLPmodel(Module):
    def __init__(self, vocab_len,embedding_dim, hidden_dim, num_class):
        super(MLPmodel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_len,embedding_dim) #dim=(21401, 32)
        self.linear1 = torch.nn.Linear(20*embedding_dim, hidden_dim) #dim=(640, 64)
        self.linear2 = torch.nn.Linear(hidden_dim, num_class) #(64, 2)
        #self.linear3 = torch.nn.Linear(32, num_class)
        self.activate = F.relu
    def forward(self, inputs): #-->input dim=(20, V)
        x = self.embedding(inputs) #-->dim=(20, 32)
        #x = x.mean(dim=1)  #'''-------> embedding完一定要壓扁，因為dim=(input_len, embedding_dim)'''
        #(X) x = x.view(-1,len(inputs)*embedding_dim) #dim=(1, 640) !!!
        #(O) 以下為處理每次batch做壓扁！！！才可以用batch為單位放進model
        x = x.view(-1, x.size(1) * x.size(2))  #x.view(-1, 20, 32)
        x = self.linear1(x) #-->dim=(1, 64)
        x = self.activate(x)
        x = self.linear2(x) #-->dim=(1, 2)
        #x = self.linear3(x)
        #x = F.log_softmax(x, dim=1) #use softmax function --> sum==100%, choose large one be output class
        return x
    '''
    MLPmodel(
    (embedding): Embedding(21401, 32)
    (linear1): Linear(in_features=640, out_features=64, bias=True)
    (linear2): Linear(in_features=64, out_features=2, bias=True)
    )
    '''

if __name__ == '__main__':

    x_train, y_train, x_test, y_test, word_list = preprocess()
    #print(x_train.shape) #(8000,20)
    #print(y_train.shape) #(8000,)
    #print(x_test.shape)  #(2662,20)
    #print(y_test.shape)  #(2662)

    # dataset and dataloader
    batch_size = 32
    train_dataset = TensorDataset(x_train,y_train)
    test_dataset = TensorDataset(x_test, y_test)

    #data in batch : each batch including "32 reviews" and "32 result"
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)


    device = torch.device("cpu")
    embedding_dim = 32
    hidden_dim = 64
    num_class = 2
    epoch_num = 100

    from tqdm.auto import tqdm
    model = MLPmodel(vocab_len=len(word_list), embedding_dim=embedding_dim, hidden_dim=hidden_dim,num_class=num_class)
    criteria = torch.nn.CrossEntropyLoss()   
    optim = torch.optim.SGD(model.parameters(),lr=0.02)  #SGD比Adam好誒
    print(model)

    losses = []
    for epoch in range(epoch_num):
        index = 0
        total_loss = 0
        
        for batch in tqdm(train_dataloader,desc="training : "):
            x,y=batch[0].long().to(device),batch[1].long().to(device) #y.shape=(32)
            logits = model(x)  #logits.shape=(32,2) cause two classes!!!   
            loss = criteria(logits,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        losses.append(total_loss)
        print(f"epoch = {epoch+1}, total loss : ",total_loss)

    plt.title("loss for epoches")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(losses)
    plt.show()
    torch.save(model.state_dict(),'nn/models/my_model.pth')


    correct = 0
    device=torch.device("cpu")
    total = 0  
    with torch.no_grad():
        for batch in train_dataloader:
            inputs,labels=batch[0].long().to(device),batch[1].long().to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    accuracy = correct / total
    print("accuracy = ",accuracy)

