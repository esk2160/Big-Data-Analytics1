#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import nltk
from nltk import FreqDist


# In[13]:


raw_data = f = open('Trumptweets.csv')
raw = f.read()


# In[14]:


type(raw)


# In[ ]:





# In[15]:


tokens = nltk.word_tokenize(raw)
type(tokens)


# In[16]:


words1 = [w.lower() for w in tokens]   #list comprehension 

#only keep text words, no numbers 
words2 = [w for w in words1 if w.isalpha()]


# In[17]:


freq = FreqDist(words2)
sorted_freq = sorted(freq.items(),key = lambda k:k[1], reverse = True)
sorted_freq


# In[31]:


freq.plot(30)


# In[32]:


from nltk.corpus import stopwords
stopwords = stopwords.words('english')


# In[33]:


words_nostopwords = [w for w in words2 if w not in stopwords]


# In[34]:


freq_nostw = FreqDist(words_nostopwords)

sorted_freq_nostw = sorted(freq_nostw.items(),key = lambda k:k[1], reverse = True)
sorted_freq_nostw


# In[35]:


freq_nostw.plot(30)


# In[20]:


POS_tags = nltk.pos_tag(tokens) #use unprocessed 'tokens', not 'words'
POS_tags


# In[21]:


POS_tag_listN = [(word,tag) for (word,tag) in POS_tags if tag.startswith('N')]


# In[22]:


len(POS_tag_listN)


# In[23]:


#Generate a frequency distribution 
tag_freq = nltk.FreqDist(POS_tag_listN)
sorted_tag_freq = sorted(tag_freq.items(), key = lambda k:k[1], reverse = True)
sorted_tag_freq


# In[25]:


tag_freq.plot(30)


# In[ ]:





# In[ ]:





# In[36]:


import nltk # be sure to have stopwords installed for this using nltk.download_shell()
import pandas as pd 
import string
messages = [line.rstrip() for line in open("Trumptweets.csv")]
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
summary = {"positive":0,"neutral":0,"negative":0}
for x in messages: 
    ss = sid.polarity_scores(x)
    if ss["compound"] == 0.0: 
        summary["neutral"] +=1
    elif ss["compound"] > 0.0:
        summary["positive"] +=1
    else:
        summary["negative"] +=1
print(summary)


# In[37]:


import matplotlib.pyplot as plt
labels = "positive", "neutral", "negative"
sizes= [81, 20, 38]
colors = ['gold', 'yellowgreen', 'lightcoral']
# Plot
plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()


# In[ ]:




