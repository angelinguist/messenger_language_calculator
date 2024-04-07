# PRELIMINARIES #################################################################
import json
import re
import demoji
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# FUNCTIONS ####################################################################

## LANGUAGE DETECTION ##########################################################
model_ckpt = "papluca/xlm-roberta-base-language-detection"
pipe = pipeline("text-classification", model=model_ckpt)
# check if a word is a word in polish or english or neither
def lg_checker(text):
    res = pipe(text, top_k=1, truncation=True)
    res = res[0]['label']
    if res in ['pl', 'en']:
        return res
    else:
        return 'neither'

## NAMED ENTITY RECOGNITION ####################################################

tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

def ner_check(text):
  res = nlp(text)
  return res

# DATA CLEANING ################################################################

with open('message_1.json', 'r', encoding='utf-8') as file:
    data = json.load(file) #dictionary with two entries: participants and messages

participants = data['participants'] #participants dictionary (len=1)
messages = data['messages'][0:30] #messages dictionary (len=1, has a list of dictionaries inside)

for item in messages: #clean up the polish signs in messages and sender names, delete data that won't be used
    del item['timestamp_ms']
    del item['is_geoblocked_for_viewer']
    if 'reactions' in item:
        del item['reactions']
    if 'photos' in item:
        del item['photos']
    if 'sticker' in item:
        del item['sticker']
    if 'share' in item:
        del item['share']
    if 'videos' in item:
        del item['videos']
    item['sender_name'] = item['sender_name'].encode('latin-1').decode('utf-8')
    if 'content' in item:
        item['content']= item['content'].encode('latin-1').decode('utf-8') #decode it properly
        item['content'] = item['content'].lower() #lowercase
        item['content'] = item['content'].replace('\n', '') #get rid of newline characters
        item['content'] = demoji.replace(item['content'], "") #get rid of emojis

# LANGUAGE STATISTICS ##########################################################
cache_pl=[] #caches to avoid checking words with a pipeline if they've already been categorized
cache_en=[]

listofnames =[] #list of all participants
for item in participants:
    item['name'] = item['name'].encode('latin-1').decode('utf-8') #fix polish signs
    listofnames.append(item['name'])

lgstats = {} #dictionary to store how many words in Pl, En, other
for name in listofnames:
    lgstats[name] = [0,0,0] #pl[0], eng[1], other/undefined[2]

named_entities = {} #dictionary to store named entities

for item in messages:
    if 'content' in item:
      deconstructed = item['content'].split()
      reconstructed =[] #put the message back together with fixed multiple signs
      for word in deconstructed:
        if word not in cache_pl and word not in cache_en: #check cache first
          checked = lg_checker(word) #then use the pipeline
          if checked == 'pl':
            lgstats[item['sender_name']][0]+=1
            reconstructed.append(word)
            cache_pl.append(word)
          elif checked == 'en':
            lgstats[item['sender_name']][1]+=1
            reconstructed.append(word)
            cache_en.append(word)
          else:
            word = re.sub('(.)\\1*', '\\1',word) #subs multiple chars for one (would probably need to sub for two, check, then for one)
            reconstructed.append(word)
            if word in cache_pl:
              lgstats[item['sender_name']][0]+=1
            elif word in cache_en:
              lgstats[item['sender_name']][1]+=1
            else:
              checked_again = lg_checker(word)
              if checked_again =='pl':
                lgstats[item['sender_name']][0]+=1
                cache_pl.append(word)
              elif checked_again == 'en':
                lgstats[item['sender_name']][1]+=1
                cache_en.append(word)
              else:
                lgstats[item['sender_name']][2]+=1
        elif word in cache_pl:
          lgstats[item['sender_name']][0]+=1
          reconstructed.append(word)
        elif word in cache_en:
          lgstats[item['sender_name']][1]+=1
          reconstructed.append(word)
      item['content'] = ' '.join(reconstructed) #reconstruct message
      out = ner_check(item['content']) #check whole reconstructed message for named entities
      for dic in out: #output of ner_check is a list of dictionaries
        k = dic['word'] #key will be the value for word in them
        if k not in named_entities.keys(): #add if doesn't exist
          named_entities[k]=1
        else:
          k =+1 #if exists, add count

print(lgstats) #raw stats
print(named_entities) #not a good model it seems, but only multilingual one I found

# FINAL SUMMARY OUTPUT #########################################################

for item in lgstats: #print pretty stats with percentages calculated
    print(f'{item} used {lgstats[item][0]} Polish words, {lgstats[item][1]} English words, and {lgstats[item][2]} undefined words \nThis means that {item} used {round((lgstats[item][0]/(lgstats[item][0]+lgstats[item][1]+lgstats[item][2])*100),2)}% Polish words, {round((lgstats[item][1]/(lgstats[item][0]+lgstats[item][1]+lgstats[item][2])*100),2)}% English words, and {round((lgstats[item][2]/(lgstats[item][0]+lgstats[item][1]+lgstats[item][2])*100),2)}% words that could not be identified as either Polish or English')