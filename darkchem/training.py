from darkchem.utils import test_train_split, DataGenerator, SMI, savedict
import numpy as np
import pandas as pd
import os


def train(args):
    import keras
    from darkchem.network import VAE
    from darkchem.callbacks import MultiModelCheckpoint

    # load data
    x = np.load(args.data).astype(np.uint8)
    n, m = x.shape
    d = max(np.unique(x)) + 1

    # test/train split
    mask = test_train_split(x, args.validation)
    x_train = x[mask]
    x_validation = x[~mask]

    # optionally load labels
    if args.labels != '-1':
        labels = np.load(args.labels)
        labels_train = labels[mask]
        labels_validation = labels[~mask]
        args.nlabels = labels.shape[-1]

    args.max_length = m
    args.nchars = d
    savedict(vars(args), args.output, verbose=True)

    # one-hot encode targets
    y_validation = keras.utils.to_categorical(x_validation, d)
    y_validation = y_validation.reshape((-1, m, d))

    y_train = keras.utils.to_categorical(x_train, d)
    y_train = y_train.reshape((-1, m, d))

    # initialize autoencoder
    model = VAE()

    # multitask
    if args.labels != '-1':
        model.create_multitask(nchars=d, max_length=m, kernels=args.kernels, filters=args.filters,
                               embedding_dim=args.embedding_dim, latent_dim=args.latent_dim, epsilon_std=args.epsilon,
                               nlabels=labels.shape[-1], dropout=args.dropout, freeze_vae=args.freeze_vae)

        # model checkpointing
        models = [model.autoencoder, model.encoder, model.encoder_variational, model.decoder, model.predictor, model.embedder]
        filepaths = [os.path.join(args.output, f) for f in ('vae.h5',
                                                            'encoder.h5',
                                                            'encoder+v.h5',
                                                            'decoder.h5',
                                                            'predictor.h5',
                                                            'embedder.h5')]
        checkpoint = MultiModelCheckpoint(models, filepaths, monitor='val_loss',
                                          save_best_only=True, mode='min', save_weights_only=True)

    # vae only
    else:
        model.create(nchars=d, max_length=m, kernels=args.kernels, filters=args.filters,
                     embedding_dim=args.embedding_dim, latent_dim=args.latent_dim, epsilon_std=args.epsilon,
                     freeze_vae=args.freeze_vae)

        # model checkpointing
        models = [model.autoencoder, model.encoder, model.encoder_variational, model.decoder, model.embedder]
        filepaths = [os.path.join(args.output, f) for f in ('vae.h5',
                                                            'encoder.h5',
                                                            'encoder+v.h5',
                                                            'decoder.h5',
                                                            'embedder.h5')]
        checkpoint = MultiModelCheckpoint(models, filepaths, monitor='val_loss',
                                          save_best_only=True, mode='min', save_weights_only=True)

    # print model summary
    print(model.autoencoder.summary())

    # optionally load weights
    if args.weights != '-1':
        model.encoder.load_weights(os.path.join(args.weights, 'encoder.h5'))
        model.encoder_variational.load_weights(os.path.join(args.weights, 'encoder+v.h5'))
        model.decoder.load_weights(os.path.join(args.weights, 'decoder.h5'))

        # try to load predictor weights
        if (args.labels != '-1') and (os.path.exists(os.path.join(args.weights, 'predictor.h5'))):
            model.predictor.load_weights(os.path.join(args.weights, 'predictor.h5'))

    # early stopping
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, mode='min')

    # train multitask
    if args.labels != '-1':
        model.autoencoder.fit(x_train, [y_train, labels_train],
                              batch_size=args.batch_size,
                              epochs=args.epochs,
                              validation_data=(x_validation, [y_validation, labels_validation]),
                              callbacks=[early_stop, checkpoint],
                              shuffle=True,
                              verbose=2)
    # train vae
    else:
        model.autoencoder.fit(x_train, y_train,
                              batch_size=args.batch_size,
                              epochs=args.epochs,
                              validation_data=(x_validation, y_validation),
                              callbacks=[early_stop, checkpoint],
                              shuffle=True,
                              verbose=2)


def train_generator(args, charset=SMI):
    import keras
    from darkchem.network import VAE
    from darkchem.callbacks import MultiModelCheckpoint

    # test/train split
    files = pd.read_csv(os.path.join(args.data, 'index.tsv'), sep='\t', header=None)
    idx = np.arange(len(files.index))

    # load first partition for metadata
    x = np.load(os.path.join(args.data, files.iloc[0, 0]))
    n, m = x.shape

    args.max_length = m
    args.nchars = len(charset)
    savedict(vars(args), args.output, verbose=True)

    # shuffle data
    np.random.shuffle(idx)

    # validation set size
    nvalidation = int(args.validation * len(files.index))

    # test/train split
    train_examples = files.iloc[idx[nvalidation:], -1].sum()
    validation_examples = files.iloc[idx[:nvalidation], -1].sum()
    partition = {'train': [os.path.join(args.data, x) for x in files.iloc[idx[nvalidation:], 0]],
                 'validation': [os.path.join(args.data, x) for x in files.iloc[idx[:nvalidation], 0]]}

    if args.labels != '-1':
        # load first partition for metadata
        y = np.load(os.path.join(args.data, files.iloc[0, 1]))
        nlabels = y.shape[-1]

        labels = {'train': [os.path.join(args.data, x) for x in files.iloc[idx[nvalidation:], 1]],
                  'validation': [os.path.join(args.data, x) for x in files.iloc[idx[:nvalidation], 1]]}
    else:
        labels = {'train': None,
                  'validation': None}

    # intialize generators
    training_generator = DataGenerator(charset=charset,
                                       batch_size=args.batch_size,
                                       shuffle=True).generate(partition['train'], labels['train'])
    validation_generator = DataGenerator(charset=charset,
                                         batch_size=args.batch_size,
                                         shuffle=True).generate(partition['validation'], labels['validation'])

    # initialize autoencoder
    model = VAE()

    # multitask
    if args.labels != '-1':
        model.create_multitask(nchars=len(charset), max_length=m, kernels=args.kernels, filters=args.filters,
                               embedding_dim=args.embedding_dim, latent_dim=args.latent_dim, epsilon_std=args.epsilon,
                               nlabels=nlabels, dropout=args.dropout, freeze_vae=args.freeze_vae)

        # model checkpointing
        models = [model.autoencoder, model.encoder, model.encoder_variational, model.decoder, model.predictor, model.embedder]
        filepaths = [os.path.join(args.output, f) for f in ('vae.h5',
                                                            'encoder.h5',
                                                            'encoder+v.h5',
                                                            'decoder.h5',
                                                            'predictor.h5',
                                                            'embedder.h5')]
        checkpoint = MultiModelCheckpoint(models, filepaths, monitor='val_loss',
                                          save_best_only=True, mode='min', save_weights_only=True)

    # vae only
    else:
        model.create(nchars=len(charset), max_length=m, kernels=args.kernels, filters=args.filters,
                     embedding_dim=args.embedding_dim, latent_dim=args.latent_dim, epsilon_std=args.epsilon,
                     freeze_vae=args.freeze_vae)

        # model checkpointing
        models = [model.autoencoder, model.encoder, model.encoder_variational, model.decoder, model.embedder]
        filepaths = [os.path.join(args.output, f) for f in ('vae.h5',
                                                            'encoder.h5',
                                                            'encoder+v.h5',
                                                            'decoder.h5',
                                                            'embedder.h5')]
        checkpoint = MultiModelCheckpoint(models, filepaths, monitor='val_loss',
                                          save_best_only=True, mode='min', save_weights_only=True)

    # print model summary
    print(model.autoencoder.summary())

    # optionally load weights
    if args.weights != '-1':
        model.encoder.load_weights(os.path.join(args.weights, 'encoder.h5'))
        model.encoder_variational.load_weights(os.path.join(args.weights, 'encoder+v.h5'))
        model.decoder.load_weights(os.path.join(args.weights, 'decoder.h5'))

        # try to load predictor weights
        if (args.labels != '-1') and (os.path.exists(os.path.join(args.weights, 'predictor.h5'))):
            model.predictor.load_weights(os.path.join(args.weights, 'predictor.h5'))

    # early stopping
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, mode='min')

    # train
    model.autoencoder.fit_generator(generator=training_generator,
                                    steps_per_epoch=train_examples // args.batch_size,
                                    validation_data=validation_generator,
                                    validation_steps=validation_examples // args.batch_size,
                                    epochs=args.epochs,
                                    callbacks=[early_stop, checkpoint],
                                    shuffle=True,
                                    verbose=2)
