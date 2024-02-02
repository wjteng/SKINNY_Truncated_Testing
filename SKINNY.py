import numpy as np
from os import urandom

round_constants =  np.array([0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33,
                                  0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B,
                                  0x17, 0x2E, 0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29,
                                  0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13, 0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a,
                                  0x15, 0x2a, 0x14, 0x28, 0x10, 0x20])

sbox4 =  np.array([12, 6, 9, 0, 1, 10, 2, 11, 3, 8, 5, 13, 4, 14, 7, 15])
sbox4_inv =  np.array([3, 4, 6, 8, 12, 10, 1, 14, 9, 2, 5, 7, 0, 11, 13, 15])

lookup_TK2 = np.array([0,2,4,6,9,11,13,15,1,3,5,7,8,10,12,14])
lookup_TK3 = np.array([0,8,1,9,2,10,3,11,12,4,13,5,14,6,15,7])

def WORD_SIZE():
    return(64);

def tweakey_permute(tk) :
    temp = np.zeros_like(tk)
    temp[:,2:4] = tk [:,0:2]

    val_8 = tk[:,2] >> 12
    val_9 = (tk[:,2] >> 8) & 0xf
    val_10 = (tk[:,2] >> 4) & 0xf
    val_11 = tk[:,2] & 0xf

    val_12 = tk[:,3] >> 12
    val_13 = (tk[:,3] >> 8) & 0xf
    val_14 = (tk[:,3] >> 4) & 0xf
    val_15 = tk[:,3] & 0xf

    temp[:,0] = (val_9 << 12) + (val_15 << 8) + (val_8 << 4) + val_13
    temp[:,1] = (val_10 << 12) + (val_14 << 8) + (val_12 << 4) + val_11

    return temp

def LFSR2(tk):
    val_0 = tk[:,0] >> 12
    val_1 = (tk[:,0] >> 8) & 0xf
    val_2 = (tk[:,0] >> 4) & 0xf
    val_3 = tk[:,0] & 0xf

    val_4 = tk[:,1] >> 12
    val_5 = (tk[:,1] >> 8) & 0xf
    val_6 = (tk[:,1] >> 4) & 0xf
    val_7 = tk[:,1] & 0xf

    tk[:,0] = (lookup_TK2[val_0] << 12) + (lookup_TK2[val_1]<< 8) + (lookup_TK2[val_2] << 4) + lookup_TK2[val_3]
    tk[:,1] = (lookup_TK2[val_4] << 12) + (lookup_TK2[val_5]<< 8) + (lookup_TK2[val_6] << 4) + lookup_TK2[val_7]
    return tk

def LFSR3(tk):
    val_0 = tk[:,0] >> 12
    val_1 = (tk[:,0] >> 8) & 0xf
    val_2 = (tk[:,0] >> 4) & 0xf
    val_3 = tk[:,0] & 0xf

    val_4 = tk[:,1] >> 12
    val_5 = (tk[:,1] >> 8) & 0xf
    val_6 = (tk[:,1] >> 4) & 0xf
    val_7 = tk[:,1] & 0xf

    tk[:,0] = (lookup_TK3[val_0] << 12) + (lookup_TK3[val_1]<< 8) + (lookup_TK3[val_2] << 4) + lookup_TK3[val_3]
    tk[:,1] = (lookup_TK3[val_4] << 12) + (lookup_TK3[val_5]<< 8) + (lookup_TK3[val_6] << 4) + lookup_TK3[val_7]
    return tk

def expand_keys(keys,nr,t) :
    ks = np.zeros((n,nr,2),dtype=np.uint16)
    tk1 = tk2 = tk3 = keys [:,0:4]
    if t > 1:
      tk2 = keys [:,4:8]
    if t > 2:
      tk3 = keys [:,8:12]

    for i in range (0,nr):
      ks[:,i,:]=tk1[:,0:2]
      if t > 1 :
        ks[:,i,:]=ks[:,i,:] ^ tk2[:,0:2]
      if t > 2 :
        ks[:,i,:]=ks[:,i,:] ^ tk3[:,0:2]

      tk1 = tweakey_permute(tk1);
      tk2 = tweakey_permute(tk2);
      tk3 = tweakey_permute(tk3);

      if t > 1:
        tk2 = LFSR2(tk2)
      if t > 2:
        tk3 = LFSR3(tk3)

    return ks

def enc_one_round(c,k,rc):
    temp = c
    in_state = np.zeros((n,16),dtype=np.uint16)
    
    # splitting c(4 X 16-bits) into in_state(16 X 4-bits)
    for i in range(0,4):
      in_state[:,i*4+0] = c[:,i] >> 12
      in_state[:,i*4+1] = (c[:,i] >> 8) & 0xf
      in_state[:,i*4+2] = (c[:,i] >> 4) & 0xf
      in_state[:,i*4+3] = c[:,i] & 0xf

    # SubCells
    for i,x in enumerate(in_state):
      for j,y in enumerate(x):
        in_state[i,j] = sbox4[y]

    # AddCconstants
    c0 = rc & 0xF
    c1 = rc >> 4
    c2 = 0x2
    in_state[:,0] ^= c0
    in_state[:,4] ^= c1
    in_state[:,8] ^= c2
 
    # AddRoundTweakey
    for i in range(0,2):
      in_state[:,i*4+0] = (k[:,i] >> 12) ^ in_state[:,i*4+0]
      in_state[:,i*4+1] = ((k[:,i] >> 8) & 0xf) ^ in_state[:,i*4+1]
      in_state[:,i*4+2] = ((k[:,i] >> 4) & 0xf) ^ in_state[:,i*4+2]
      in_state[:,i*4+3] = (k[:,i] & 0xf) ^ in_state[:,i*4+3]
    
    # ShiftRows
    in_state = np.concatenate(( in_state[:,0:4],
                                in_state[:,7].reshape(-1,1), in_state[:,4:7],
                                in_state[:,10:12], in_state[:,8:10],
                                in_state[:,13:16], in_state[:,12].reshape(-1,1) ), axis = 1)

    # MixColumns
    mix1 = in_state[:,4:8] ^ in_state[:,8:12]
    mix2 = in_state[:,0:4] ^ in_state[:,8:12]
    mix3 = mix2 ^ in_state[:,12:16]
    in_state = np.concatenate(( mix3, in_state[:,0:4],mix1, mix2 ), axis = 1)
    
    # convert back 4 X 16-bits
    for i in range(0,4):
      temp[:,i] = (in_state[:,i*4+0] << 12) + (in_state[:,i*4+1]<< 8) + (in_state[:,i*4+2] << 4) + in_state[:,i*4+3]

    return temp



def encrypt(p, ks):
    c = p.copy()
    round_num = 1

    for k in ks:
      c = enc_one_round(c,k,round_constants[round_num-1])
      round_num += 1

    return c

# UNDEFINED
#def decrypt(c, ks):
#    x, y = c[0], c[1];
#    for k in reversed(ks):
#        x, y = dec_one_round((x,y), k);
#    return(x,y);



#convert_to_binary takes as input an array of ciphertext pairs
#where the first row of the array contains the lefthand side of the ciphertexts,
#the second row contains the righthand side of the ciphertexts,
#the third row contains the lefthand side of the second ciphertexts,
#and so on
#it returns an array of bit vectors containing the same data
def convert_to_binary(arr):
  X = np.zeros((2 * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
  for i in range(2 * WORD_SIZE()):
    index = i // WORD_SIZE();
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);
  
## THIS IS FROM SPECK ##
#takes a text file that contains encrypted block0, block1, true diff prob, real or random
#data samples are line separated, the above items whitespace-separated
#returns train data, ground truth, optimal ddt prediction
#def readcsv(datei):
#    data = np.genfromtxt(datei, delimiter=' ', converters={x: lambda s: int(s,16) for x in range(2)});
#    X0 = [data[i][0] for i in range(len(data))];
#    X1 = [data[i][1] for i in range(len(data))];
#    Y = [data[i][3] for i in range(len(data))];
#    Z = [data[i][2] for i in range(len(data))];
#    ct0a = [X0[i] >> 16 for i in range(len(data))];
#    ct1a = [X0[i] & MASK_VAL for i in range(len(data))];
#    ct0b = [X1[i] >> 16 for i in range(len(data))];
#    ct1b = [X1[i] & MASK_VAL for i in range(len(data))];
#    ct0a = np.array(ct0a, dtype=np.uint16); ct1a = np.array(ct1a,dtype=np.uint16);
#    ct0b = np.array(ct0b, dtype=np.uint16); ct1b = np.array(ct1b, dtype=np.uint16);
    
#    #X = [[X0[i] >> 16, X0[i] & 0xffff, X1[i] >> 16, X1[i] & 0xffff] for i in range(len(data))];
#    X = convert_to_binary([ct0a, ct1a, ct0b, ct1b]); 
#    Y = np.array(Y, dtype=np.uint8); Z = np.array(Z);
#    return(X,Y,Z);

#baseline training data generator
def make_train_data(n, nr, diff=(0x0040,0)):
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  num_rand_samples = np.sum(Y==0);
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  ks = expand_key(keys, nr);
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
  return(X,Y);

#real differences data generator
def real_differences_data(n, nr, diff=(0x0040,0)):
  #generate labels
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  #generate keys
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  #generate plaintexts
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  #apply input difference
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  num_rand_samples = np.sum(Y==0);
  #expand keys and encrypt
  ks = expand_key(keys, nr);
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  #generate blinding values
  k0 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  k1 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  #apply blinding to the samples labelled as random
  ctdata0l[Y==0] = ctdata0l[Y==0] ^ k0; ctdata0r[Y==0] = ctdata0r[Y==0] ^ k1;
  ctdata1l[Y==0] = ctdata1l[Y==0] ^ k0; ctdata1r[Y==0] = ctdata1r[Y==0] ^ k1;
  #convert to input data for neural networks
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
  return(X,Y);

