import SKINNY as cipher


import numpy as np

from pickle import dump

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
#from keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2
from keras import callbacks

bs = 5000;
wdir = './freshly_trained_nets/'

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);

def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
  return(res);

#make residual tower of convolutional blocks
def make_resnet(num_blocks=1, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=64, ks=3,depth=5, reg_param=0.0001, final_activation='sigmoid',inactive_count=16):
  print("No. Cells: ",inactive_count)
  #Input and preprocessing layers
  #inp = Input(shape=(num_blocks * word_size * 2,));
  inp = Input(shape=(inactive_count*4*2,))
  print(inp.shape)
  #rs = Reshape((2 * num_blocks, word_size))(inp);
  #rs = Reshape((active_count,4))(inp);
  rs = Reshape((2,inactive_count*4))(inp)
  num_filters = inactive_count*2*4
  if num_filters % 64 ==0:
    num_filters = 64
  print(rs.shape)
  perm = Permute((2,1))(rs);
  print(perm.shape)
  #add a single residual layer that will expand the data to num_filters channels
  #this is a bit-sliced layer
  conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm);
  conv0 = BatchNormalization()(conv0);
  conv0 = Activation('relu')(conv0);
  print(conv0.shape)
  #add residual blocks
  shortcut = conv0;
  for i in range(depth):
    conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut);
    conv1 = BatchNormalization()(conv1);
    conv1 = Activation('relu')(conv1);
    conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1);
    conv2 = BatchNormalization()(conv2);
    conv2 = Activation('relu')(conv2);
    shortcut = Add()([shortcut, conv2]);
    print(shortcut.shape) 
  #add prediction head
  flat1 = Flatten()(shortcut);
  dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1);
  dense1 = BatchNormalization()(dense1);
  dense1 = Activation('relu')(dense1);
  dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
  dense2 = BatchNormalization()(dense2);
  dense2 = Activation('relu')(dense2);
  out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2);
  model = Model(inputs=inp, outputs=out);
  return(model);

def train_distinguisher(num_epochs,diff = (0,0,0,0x0001), num_rounds=7, depth=1,trunc=(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)):
    inactive_count = 0;
    for i in trunc:
      if i == 0:
        inactive_count += 1
    
    print("inactive_count:",inactive_count)
    print("active_count:",(16-inactive_count))
    #create the network

    #learn from active only
    net = make_resnet(depth=depth, reg_param=10**-5,inactive_count=16-inactive_count); #HERE
    net.compile(optimizer='adam',loss='mse',metrics=['acc']);
    
    #generate training and validation data (make train data)
    #X, Y = cipher.make_train_data(10**7,num_rounds,diff);
    #X_eval, Y_eval = cipher.make_train_data(10**6, num_rounds,diff);
    
    #generate training and validation data (real difference)
    X, Y = cipher.real_differences_data(10**6,num_rounds,diff,1);
    print(X.shape)
    for index, value in enumerate(reversed(trunc)) :
      # removing the inactive nibbles, leaving only active nibbles
      if value == 0:
        X = np.delete(X,slice((15-index)*4+64,(15-index)*4+4+64),1)
    for index, value in enumerate(reversed(trunc)) :
      # removing the inactive nibbles, leaving only active nibbles
      if value == 0:
        X = np.delete(X,slice((15-index)*4,(15-index)*4+4),1)

        
    X_eval, Y_eval = cipher.real_differences_data(10**5, num_rounds,diff,1);
    for index, value in enumerate(reversed(trunc)) :
      # removing the inactive nibbles, leaving only active nibbles
      if value == 0:
        X_eval = np.delete(X_eval,slice((15-index)*4+64,(15-index)*4+4+64),1)
    for index, value in enumerate(reversed(trunc)) :
      # removing the inactive nibbles, leaving only active nibbles
      if value == 0:
        X_eval = np.delete(X_eval,slice((15-index)*4,(15-index)*4+4),1)
  
    #set up model checkpoint
    check = make_checkpoint(wdir+'best'+str(num_rounds)+'depth'+str(depth)+'.h5');
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));
    
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  mode ="min", patience = 5, restore_best_weights = True);
                                        
    #train and evaluate
    h = net.fit(X,Y,epochs=num_epochs,batch_size=bs,validation_data=(X_eval, Y_eval), callbacks=[lr,check]);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_acc']);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_loss']);
    dump(h.history,open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth)+'.p','wb'));
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    return(net, h);
