from load_data import load_data
from model import create_model
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from config import *
from sklearn.metrics import roc_auc_score
import pickle

checkpoint_filepath = 'ml/tmp/checkpoint'

training_finished = False
x_train, x_val, x_test, y_train, y_val, y_test, vocab_size, word_index = load_data()

class scoreTarget(tf.keras.callbacks.Callback):
  def __init__(self, target):
    super(scoreTarget, self).__init__()
    self.target = target

  def on_epoch_end(self, epoch, logs={}):
    acc = logs['auc']
    if acc >= self.target:
      self.model.stop_training = True

class ROAUCMetrics(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        self.valid_x = val_data[0]
        self.valid_y = val_data[1]

    def on_train_begin(self, logs={}):
        self.val_aucs = []

    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.valid_x)
        val_auc = roc_auc_score(self.valid_y, pred,  average='micro')
        print('\nval-roc-auc: %s' % (str(round(val_auc,4))),end=100*' '+'\n')
        self.val_aucs.append(val_auc)
        return

roc = ROAUCMetrics(val_data=(x_val, y_val))
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5) # metric stop improving
tensorboard    = tf.keras.callbacks.TensorBoard() 
reduce_lr      = tf.keras.callbacks.ReduceLROnPlateau(patience=3)
# target         = scoreTarget(0.96)


# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     monitor=tf.keras.metrics.AUC(),
#     mode='auto')

model = create_model(vocab_size, MAXLEN, EMBEDDING_DIM, word_index)

print("Finished creating model")

optimizer = Adam(lr=1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.AUC(from_logits=True)])

history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val), callbacks = [roc,early_stopping, tensorboard, reduce_lr])
with open('ml/models/history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
model.save('ml/models/sentimentmodel2.h5')