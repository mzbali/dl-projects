
# coding: utf-8

# ## Looking at Data

# In[12]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


get_ipython().run_line_magic('pwd', '')


# In[14]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[15]:


bs = 48


# In[16]:


path = untar_data(URLs.CIFAR)


# In[17]:


path


# In[18]:


path.ls()


# In[19]:


path_test = path/'test'
path_train = path/'train'


# In[21]:


help(ImageDataBunch.from_folder)


# In[29]:


data = ImageDataBunch.from_folder(path,ds_tfms=get_transforms(do_flip=False), valid_pct=0.2, size=224).normalize(imagenet_stats)


# In[30]:


help(data.show_batch)


# In[31]:


data.show_batch(rows=3,figsize=(9,7))


# In[40]:


print(data.classes)


# ## Training

# In[32]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[33]:


learn


# In[41]:


learn.fit_one_cycle(4)


# In[57]:


learn.save('stage-1')


# ## Results

# In[58]:


help(ClassificationInterpretation.from_learner)


# In[59]:


interp = ClassificationInterpretation.from_learner(learn)


# In[60]:


interp.plot_top_losses(9,figsize=(9,7))


# In[61]:


interp.plot_confusion_matrix()


# In[62]:


interp.most_confused(min_val=9)


# ## Unfreezing, Fine_tuning, Learning Rate

# In[63]:


learn.unfreeze()


# In[64]:


learn.fit_one_cycle(1)


# In[65]:


learn.save('unfreeze-1')


# In[66]:


learn.lr_find()


# In[67]:


learn.recorder.plot()


# In[68]:


learn.unfreeze()


# In[70]:


learn.fit_one_cycle(2,max_lr=slice(1e-5,1e-3))


# In[71]:


interp = ClassificationInterpretation.from_learner(learn)


# In[74]:


interp.plot_top_losses(9,figsize=(15,13))


# In[75]:


learn.save('stage-2')


# Preety Accurate Model!!
