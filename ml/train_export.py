"""Train an educational binary SMS classifier and export a genuine .tflite file.

The messages are AUTHOR-WRITTEN templates with randomized details, not the real
UCI SMS Spam Collection. Synthetic test scores are not real-world accuracy.
Run: python ml/train_export.py (requires torch, numpy, scikit-learn).
"""
from pathlib import Path
from flatbuffer_writer import export_network
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch import nn
import torch, random, re, json, numpy as np

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'app/src/main/assets'; A.mkdir(parents=True,exist_ok=True)
SEED=20261030; rng=random.Random(SEED); torch.manual_seed(SEED)
torch.set_num_threads(1); np.random.seed(SEED)

# Fully fictional phrases to be used only for training an educational demo.
SPAM=[
 'URGENT! You have won {prize}. Claim your reward now at {url}.',
 'WINNER! You were selected for {prize}. Text CLAIM to {phone}.',
 'You qualify for a cash loan of {amount}. Apply immediately at {url}.',
 'FINAL NOTICE: Your {service} account is suspended. Verify it at {url}.',
 'Your {service} will be locked unless you confirm your password at {url}.',
 'Unclaimed {prize}! Call {phone} now to receive your money.',
 'Earn {amount} every week from home. Register on {url}.',
 'Limited offer: free {thing}, pay a small processing fee at {url}.',
 'Parcel delivery failed! Pay a redelivery fee at {url}.',
 'Your refund of {amount} is ready. Submit your bank details at {url}.',
 'Your number won a lucky draw. Text WIN to {phone} to collect {prize}.',
 'Exclusive investment scheme! Double your money by joining {url}.',
 'Claim your reward of {amount} today before the deadline expires.',
 'Your ATM card is expiring. Send your PIN and OTP to reactivate it.',
 'Special promotion! Buy a {thing} now with a huge discount on {url}.',
 'You have been selected as a lucky customer. Send your address for {prize}.',
 'Your {prize} expires tonight. Click {url} to collect it immediately.',
 'Free money! Claim {amount} now by paying an activation charge.',
 'Share your password and verification code at {url} to unlock your account.',
 'Urgent payment required. Your {service} service will stop today: {url}.',
 'Text WIN to {phone} for a chance to win a new {thing}.',
 'This is your final chance to claim {amount} in prize funds.',
 'Government grant of {amount} approved. Send your bank details now.',
 'You have an unpaid toll fee. Settle it immediately at {url}.',
 'Your delivery is on hold until you complete a payment at {url}.',
 'Congratulations, free {thing} for you! Visit {url} to get it.',
]
NORMAL=[
 'Hi {name}, let us meet near the {place} at {time} tomorrow.',
 'Reminder: our {subject} lecture begins at {time}.',
 'Could you send me the {doc} when you are free?',
 'I will bring your {thing} to the {place} after lunch.',
 'Thanks {name}, I received your {doc}. See you soon.',
 'Our {subject} group meeting starts at {time} in the {place}.',
 'Please finish the {doc} before the next class.',
 'Hello {name}, I am going home and I will call you later.',
 'The {place} closes at {time} today.',
 'I left my {thing} on a table in the {place}.',
 'The team will review the {doc} after the {subject} lecture.',
 'Please bring the {doc} to the {place} tomorrow morning.',
 'Our family dinner starts at {time} tonight.',
 'Your appointment has been booked for {time}.',
 'Your order has been delivered. Thank you for shopping.',
 'We received your message and will reply within two working days.',
 'Your login verification code is 123456. Do not share it with anyone.',
 'Your online payment receipt is attached for your records.',
 'The {subject} teacher uploaded the {doc} today.',
 'The bus arrives at {time}. I will meet you at the {place}.',
 'I completed the {subject} project and submitted my {doc}.',
 'Hey {name}, can we discuss the homework after the {subject} class?',
 'Your library books are ready for collection.',
 'I have shared our class notes at https://class-notes.example/notes.',
 'Your package will arrive tomorrow. Check your order page.',
 'Your payment was received successfully. Thank you.',
 'Please remember to charge the {thing} before the presentation.',
]
CHOICES={
 'name':['Alex','Sara','Ali','Noor','Sam','Hina'],
 'place':['campus','library','cafe','office','classroom','shop'],
 'subject':['math','Android','security','networking','database','computer science'],
 'doc':['report','slides','assignment','notes','PDF','schedule'],
 'thing':['phone','laptop','headphones','charger','book','tablet'],
 'time':['eight','nine','ten','two','three','five'],
 'prize':['cash reward','gift voucher','free phone','lottery prize','bonus payment'],
 'amount':['500','2000','10000','25000','100000'],
 'service':['bank','delivery','mobile wallet','parcel','payment','online'],
 'url':['https://claim-gift.example/redeem','https://account-check.example/login',
        'www.prize.example/claim','https://parcel-fee.example/pay'],
 'phone':['88122','55219','03001234567'],
}

def fill(template):return template.format(**{key:rng.choice(v) for key,v in CHOICES.items()})

examples=[]
for patterns,label in [(NORMAL,0),(SPAM,1)]:
    for template in patterns:
        for i in range(170):
            text=fill(template)
            if i%6==0:text=rng.choice(['Hello! ','Notice: ','Please read: ','Hi! '])+text
            if i%7==0:text+=' '+rng.choice(['Thank you.','Please reply.','Act soon.'])
            examples.append((text,label))
unique={}
for text,label in examples:unique.setdefault(text.lower(),(text,label))
texts=[x for x,y in unique.values()]
labels=np.asarray([y for x,y in unique.values()],dtype=np.int64)
train_txt,test_txt,train_y,test_y=train_test_split(texts,labels,random_state=SEED,
                                                  stratify=labels,test_size=.2)
URL=re.compile(r'(?i)(?:https?://|www\.)\S+')
NUM=re.compile(r'\b[0-9]{5,}\b')

def normalize(s):
    s=URL.sub(' urltoken ',s.lower());s=NUM.sub(' numbertoken ',s)
    for c in ('£','$','€'):s=s.replace(c,' moneytoken ')
    return s

V=CountVectorizer(binary=True,ngram_range=(1,2),min_df=2,max_features=1536,
                  token_pattern=r"(?u)\b[a-z]+(?:'[a-z]+)?\b")
X=V.fit_transform(map(normalize,train_txt)).toarray().astype('float32')
T=V.transform(map(normalize,test_txt)).toarray().astype('float32')
vocab=V.get_feature_names_out().tolist()
(A/'vocabulary.txt').write_text('\n'.join(vocab)+'\n',encoding='utf-8')
net=nn.Sequential(nn.Linear(len(vocab),40),nn.ReLU(),nn.Linear(40,2))
optimizer=torch.optim.AdamW(net.parameters(),lr=.009,weight_decay=.001)
criterion=nn.CrossEntropyLoss()
xx=torch.from_numpy(X);yy=torch.from_numpy(train_y)
for epoch in range(65):
    net.train()
    order=torch.randperm(len(yy))
    for start in range(0,len(order),192):
        ix=order[start:start+192]
        optimizer.zero_grad();loss=criterion(net(xx[ix]),yy[ix]);loss.backward();optimizer.step()
net.eval()
with torch.no_grad(): probs=net(torch.from_numpy(T)).softmax(1).numpy()
pred=probs.argmax(axis=1)
acc=float(np.mean(pred==test_y));cm=confusion_matrix(test_y,pred,labels=[0,1]).tolist()
print('Synthetic samples:',len(texts),'held-out:',len(test_y),'synthetic accuracy:',round(acc,4))
print('Confusion matrix normal/spam:',cm)
with torch.no_grad():
    w0=net[0].weight.numpy().copy();b0=net[0].bias.numpy().copy()
    w1=net[2].weight.numpy().copy();b1=net[2].bias.numpy().copy()
np.savez(ROOT/'ml/weights.npz',w0=w0,b0=b0,w1=w1,b1=b1)
blob=export_network(w0,b0,w1,b1,'Educational SpamShield neural model')
(A/'spamshield.tflite').write_bytes(blob)
CHECKS=[
 ('URGENT! You have won a cash prize. Click https://example.com/claim now!',1),
 ('Hi, the class starts at 9 tomorrow. Please bring your assignment notebook.',0),
 ('Your account will be locked. Click now to verify your password.',1),
 ('Can you send the project report to me today?',0),
 ('Your parcel is on hold. Pay a redelivery fee at www.fake.example/pay',1),
 ('I will meet you at the library after lunch.',0),
 ('You are selected to receive a cash reward. Claim immediately.',1),
 ('Our practical exam begins at nine, please arrive early.',0),
 ('Final notice: your payment is overdue, click this link now.',1),
 ('Mom I reached home safely and will call you soon.',0),
 ('Please share the Android slides for the group discussion.',0),
 ('Your OTP is 123456. Do not share it with anyone.',0),
 ('You won a free phone. Text WIN to 88122 to collect it.',1),
 ('Hi Sam, are you coming to the workshop tomorrow?',0),
 ('Get rich quickly by joining our investment scheme today.',1),
 ('Thanks for the invitation, I will join the meeting.',0),
 ('Your order has been delivered to reception.',0),
 ('Bank alert: click here or your account will be suspended.',1),
]
with torch.no_grad():
    checkX=V.transform(map(normalize,[x for x,y in CHECKS])).toarray().astype('float32')
    pp=net(torch.from_numpy(checkX)).softmax(1).numpy()
correct=int(np.sum(pp.argmax(axis=1)==[y for x,y in CHECKS]))
print('Small separate authored spot check:',correct,'/',len(CHECKS))
for (msg,y),prob in zip(CHECKS,pp):
    if np.argmax(prob)!=y:print('  Misclassified:',repr(msg),'spamScore=',round(float(prob[1]),3))
(ROOT/'ml/spot_check.json').write_text(json.dumps([
 {'text':x,'expected':'spam' if y else 'normal','spam_score':round(float(p[1]),5)}
 for (x,y),p in zip(CHECKS,pp)],indent=2)+'\n',encoding='utf-8')
meta={'project':'Spam Shield','student':'Ubaid Ullah','registration':'242201064',
 'instructor':'Sir Uzair','dataset':'Author-written, synthetic educational SMS templates; NOT UCI SMS collection',
 'model':'PyTorch-trained binary-bag-of-unigrams/bigrams -> 40 ReLU -> 2 softmax',
 'classes':['Normal','Spam'],'training_examples':len(train_y),'holdout_examples':len(test_y),
 'holdout_accuracy_with_template_overlap':round(acc,4),'synthetic_confusion_matrix':cm,
 'small_authored_spot_check_correct':correct,'small_authored_spot_check_total':len(CHECKS),
 'model_input_shape':[1,len(vocab)],'model_output_shape':[1,2],
 'input_dtype':'float32','output_dtype':'float32',
 'normalization':'URL=>urltoken; 5+ digit number=>numbertoken; currency symbols=>moneytoken; binary word/bigram presence',
 'model_size_bytes':len(blob),'export_method':'TFLite FlatBuffer built from trained neural network weights',
 'limitations':'Not a real SMS corpus, tests share authored pattern families; confidence scores not calibrated, not guaranteed to spot malicious SMS. Short English only, vulnerable to obfuscation and unseen scams.'}
(ROOT/'MODEL_CARD.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
print('TFLite bytes:',len(blob),'vocabulary features:',len(vocab))
