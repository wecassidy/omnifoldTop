from tensorflow import keras
from tensorflow.keras import layers

model_builders = {} # Mapping of model names to model callables

def get_callbacks(model_filepath=None):

    EarlyStopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, restore_best_weights=True
    )

    if model_filepath:
        #checkpoint_fp = model_filepath + '_Epoch-{epoch}'
        checkpoint_fp = model_filepath
        CheckPoint = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_fp, verbose=1, monitor='val_loss',
            save_best_only=True, save_weights_only=True
        )

        logger_fp = model_filepath+'_history.csv'
        CSVLogger = keras.callbacks.CSVLogger(
            filename=logger_fp, append=False
        )

        return [CheckPoint, CSVLogger, EarlyStopping]
    else:
        return [EarlyStopping]

def get_model(input_shape, model_name='dense_3hl', nclass=2):

    model = model_builders[model_name](input_shape, nclass)

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    model.summary()

    return model

def register_model(model_func):
    """
    Decorator to save a model function to the registry of model builders.

    Parameters
    ----------
    model_func : callable(array-like, int) -> keras.models.Model

    Returns
    -------
    callable(array-like, int) -> keras.models.Model
        The callable passed in to the decorator, unaltered.

    Examples
    --------

        @register_model
        def example_model(input_shape, nclass=2):
            ...
            return keras.models.Model(...)

    This model will be registered as `model.model_builders["example_model"]`
    """
    model_builders[model_func.__name__] = model_func
    return model_func

@register_model
def dense_small(input_shape, nclass=2):
    inputs = keras.layers.Input(input_shape)
    hl1 = keras.layers.Dense(50, activation="relu")(inputs)
    dropout = keras.layers.Dropout(0.4)(hl1)
    hl2 = keras.layers.Dense(50, activation="relu")(dropout)
    outputs = keras.layers.Dense(nclass, activation="softmax")(hl2)
    return keras.models.Model(inputs=inputs, outputs=outputs)

@register_model
def dense_3hl(input_shape, nclass=2):
    inputs = keras.layers.Input(input_shape)
    hidden_layer_1 = keras.layers.Dense(100, activation='relu')(inputs)
    hidden_layer_2 = keras.layers.Dense(100, activation='relu')(hidden_layer_1)
    hidden_layer_3 = keras.layers.Dense(100, activation='relu')(hidden_layer_2)
    outputs = keras.layers.Dense(nclass, activation='softmax')(hidden_layer_3)

    nn = keras.models.Model(inputs=inputs, outputs=outputs)

    return nn

@register_model
def dense_6hl(input_shape, nclass=2):
    inputs = keras.layers.Input(input_shape)
    hidden_layer_1 = keras.layers.Dense(100, activation='relu')(inputs)
    hidden_layer_2 = keras.layers.Dense(100, activation='relu')(hidden_layer_1)
    hidden_layer_3 = keras.layers.Dense(100, activation='relu')(hidden_layer_2)
    hidden_layer_4 = keras.layers.Dense(100, activation='relu')(hidden_layer_3)
    hidden_layer_5 = keras.layers.Dense(100, activation='relu')(hidden_layer_4)
    hidden_layer_6 = keras.layers.Dense(100, activation='relu')(hidden_layer_5)
    outputs = keras.layers.Dense(nclass, activation='softmax')(hidden_layer_6)

    nn = keras.models.Model(inputs=inputs, outputs=outputs)

    return nn

@register_model
def silly(input_shape, nclass=2):
    inputs = keras.layers.Input(input_shape)
    hl = keras.layers.Dense(1000, activation="relu")(inputs)
    for _ in range(10):
        hl = keras.layers.Dense(1000, activation="relu")(hl)
    outputs = keras.layers.Dense(nclass, activation="softmax")(hl)
    return keras.models.Model(inputs=inputs, outputs=outputs)

@register_model
def tiny(input_shape, nclass=2):
    inputs = keras.layers.Input(input_shape)
    hidden = keras.layers.Dense(5, activation="relu")(inputs)
    outputs = keras.layers.Dense(nclass, activation="softmax")(hidden)
    return keras.models.Model(inputs=inputs, outputs=outputs)

@register_model
def pfn(input_shape, nclass=2, nlatent=8):
    # https://arxiv.org/pdf/1810.05165.pdf

    # expected input_shape: (n_particles, n_features...)
    assert(len(input_shape) > 1)
    nparticles = input_shape[0]

    inputs = keras.layers.Input(input_shape, name='Input')

    latent_layers = []
    for i in range(nparticles):
        particle_input = keras.layers.Lambda(lambda x: x[:,i,:], name='Lambda_{}'.format(i))(inputs)

        # per particle map to the latent space
        Phi_1 = keras.layers.Dense(100, activation='relu', name='Phi_{}_1'.format(i))(particle_input)
        Phi_2 = keras.layers.Dense(100, activation='relu', name='Phi_{}_2'.format(i))(Phi_1)
        Phi = keras.layers.Dense(nlatent, activation='relu', name='Phi_{}'.format(i))(Phi_2)
        latent_layers.append(Phi)

    # add the latent representation
    added = keras.layers.Add()(latent_layers)

    F_1 = keras.layers.Dense(100, activation='relu', name='F_1')(added)
    F_2 = keras.layers.Dense(100, activation='relu', name='F_2')(F_1)
    F_3 = keras.layers.Dense(100, activation='relu', name='F_3')(F_2)

    # output
    outputs = keras.layers.Dense(nclass, activation='softmax', name='Output')(F_3)

    nn = keras.models.Model(inputs=inputs, outputs=outputs)

    return nn
