from datetime import datetime
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

from prepare.generator import get_generator
from net.CRNN import text_recognition_model
from callbacks import VizCallback


def train():
    # Data Generator
    train_gene, train_n_batches = get_generator(mode='train')
    val_gene, val_n_batches = get_generator(mode='val')

    # Model
    model_input, y_pred, model = text_recognition_model('train')
    test_func = K.function([model_input], [y_pred])

    # Callbacks
    # early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('checkpoint/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                       save_best_only=False, save_weights_only=True, verbose=0, mode='auto', preiod=2)
    train_viz_cb = VizCallback(test_func, train_gene.next_batch(), True, train_n_batches)
    val_viz_cb = VizCallback(test_func, val_gene.next_batch(), False, val_n_batches)

    # Model Compile
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    # Training
    start = datetime.now()
    history_training = model.fit_generator(
        generator=train_gene.next_batch(),
        steps_per_epoch=train_n_batches,
        epochs=20,
        callbacks=[train_viz_cb, val_viz_cb, train_gene, val_gene, model_checkpoint],
        validation_data=val_gene.next_batch(),
        validation_steps=val_n_batches
    )
    end = datetime.now()
    print('Time to train: ', end - start)

    # Save Model
    model.save('checkpoint/final_model.h5')
