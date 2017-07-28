from operator import mul
import itertools
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import SaveWeights
from utils import EarlyStopping, WeightsLogger
from lasagne import objectives
from lasagne.layers import InputLayer
from lasagne.layers import ReshapeLayer, DenseLayer, DropoutLayer, ElemwiseSumLayer, ConcatLayer, FlattenLayer
from lasagne.layers import Conv2DLayer, Conv3DLayer, MaxPool2DLayer, MaxPool3DLayer, Pool3DLayer, batch_norm
from layers import WeightedSumLayer
from lasagne import updates
from lasagne import nonlinearities
from lasagne.init import Constant
import objective_functions as objective_f
import numpy as np


def get_epoch_finished(name, patience):
    return [
        SaveWeights(name + 'model_weights.pkl', only_best=True, pickle=False),
        WeightsLogger(name + 'weights_log.pkl'),
        EarlyStopping(patience=patience)
    ]


def get_back_pathway(forward_pathway, multi_channel=True):
    # We create the backwards path of the encoder from the forward path
    # We need to mirror the configuration of the layers and change the pooling operators with unpooling,
    # and the convolutions with deconvolutions (convolutions with diferent padding). This definitions
    # match the values of the possible_layers dictionary
    back_pathway = ''.join(['d' if l is 'c' else 'u' for l in forward_pathway[::-1]])
    last_conv = back_pathway.rfind('d')
    final_conv = 'f' if multi_channel else 'fU'
    back_pathway = back_pathway[:last_conv] + final_conv + back_pathway[last_conv + 1:]

    return back_pathway


def get_layers_greenspan(
        input_channels,
):
    input_shape = (None, input_channels, 32, 32)
    images = ['axial', 'coronal', 'sagital']
    baseline = [InputLayer(name='\033[30mbaseline_%s\033[0m' % i, shape=input_shape) for i in images]
    followup = [InputLayer(name='\033[30mfollow_%s\033[0m' % i, shape=input_shape) for i in images]
    lnets = [get_lnet(b, f, i) for b, f, i in zip(baseline, followup, images)]
    union = ConcatLayer(
        incomings=lnets,
        name='concat_final'
    )
    dense = DenseLayer(
        incoming=union,
        name='\033[32mdense_final\033[0m',
        num_units=16,
        nonlinearity=nonlinearities.very_leaky_rectify
    )
    softmax = DenseLayer(
        incoming=dense,
        name='\033[32mclass_out\033[0m',
        num_units=2,
        nonlinearity=nonlinearities.softmax
    )
    return softmax


def get_convolutional_longitudinal(
    convo_blocks,
    input_shape,
    images,
    convo_size,
    pool_size,
    number_filters,
    padding,
    drop
):
    if not isinstance(convo_size, list):
        convo_size = [convo_size] * convo_blocks

    if not isinstance(number_filters, list):
        number_filters = [number_filters] * convo_blocks
    input_shape_single = tuple(input_shape[:1] + (1,) + input_shape[2:])
    channels = input_shape[1]
    if not images:
        images = ['im%d' % i for i in range(channels / 2)]
    baseline = [InputLayer(name='\033[30mbaseline_%s\033[0m' % i, shape=input_shape_single) for i in images]
    followup = [InputLayer(name='\033[30mfollow_%s\033[0m' % i, shape=input_shape_single) for i in images]

    sub_counter = itertools.count()
    convo_counters = [itertools.count() for _ in images]
    subconvo_counters = [itertools.count() for _ in images]

    subtraction = [WeightedSumLayer(
        name='subtraction_init_%s' % i,
        incomings=[p1, p2]
    ) for p1, p2, i in zip(baseline, followup, images)]

    for c, f in zip(convo_size, number_filters):
        baseline, followup = zip(*[get_shared_convolutional_block(
            p1,
            p2,
            c,
            f,
            pool_size,
            drop,
            padding,
            sufix=i,
            counter=convo_counter
        ) for p1, p2, i, convo_counter in zip(baseline, followup, images, convo_counters)])
        index = sub_counter.next()
        subtraction = [ElemwiseSumLayer(
            name='subtraction_%s_%d' % (i, index),
            incomings=[
                get_convolutional_block(
                    s,
                    c,
                    f,
                    pool_size,
                    drop,
                    padding,
                    sufix=i,
                    counter=counter
                ),
                WeightedSumLayer(
                    name='wsubtraction_%s_%d' % (i, index),
                    incomings=[p1, p2]
                )
            ]
        ) for p1, p2, s, i, counter in zip(baseline, followup, subtraction, images, subconvo_counters)]

    return baseline, followup, subtraction


def get_layers_longitudinal(
        convo_blocks,
        input_shape,
        images=None,
        convo_size=3,
        pool_size=2,
        dense_size=256,
        number_filters=32,
        padding='valid',
        drop=0.5,
        register=False,
):
    baseline, followup, subtraction = get_convolutional_longitudinal(
        convo_blocks,
        input_shape,
        images,
        convo_size,
        pool_size,
        number_filters,
        padding,
        drop,
        register
    )

    image_union = [ConcatLayer(
        incomings=[FlattenLayer(b), FlattenLayer(s)],
        name='union_%s' % i
    ) for b, s, i in zip(baseline, subtraction, images)]

    dense = [DenseLayer(
        incoming=u,
        name='\033[32mdense_%s\033[0m' % i,
        num_units=dense_size,
        nonlinearity=nonlinearities.softmax
    ) for u, i in zip(image_union, images)]

    union = ConcatLayer(
        incomings=dense,
        name='union'
    )

    soft = DenseLayer(
        incoming=union,
        name='\033[32mclass_out\033[0m',
        num_units=2,
        nonlinearity=nonlinearities.softmax
    )

    return soft


def get_layers_longitudinal_deformation(
            convo_blocks,
            input_shape,
            d_off=1,
            images=None,
            convo_size=3,
            pool_size=2,
            dense_size=256,
            number_filters=32,
            padding='valid',
            drop=0.5,
            register=False,
):
    if not isinstance(convo_size, list):
        convo_size = [convo_size] * convo_blocks
    if not isinstance(number_filters, list):
        number_filters = [number_filters] * convo_blocks

    baseline, followup, subtraction = get_convolutional_longitudinal(
        convo_blocks,
        input_shape,
        images,
        convo_size,
        pool_size,
        number_filters,
        padding,
        drop,
        register
    )

    defo_input_shape = (input_shape[:1] + (3,) + (convo_blocks*2+d_off, convo_blocks*2+d_off, convo_blocks*2+d_off))
    deformation = [InputLayer(name='\033[30mdeformation_%s\033[0m' % i, shape=defo_input_shape) for i in images]

    defo_counters = [itertools.count() for _ in images]
    for c, f in zip(convo_size, number_filters):
        deformation = [get_convolutional_block(
            d,
            c,
            f,
            pool_size,
            drop,
            padding,
            sufix='d' + i,
            counter=defo_counter
        ) for d, i, defo_counter in zip(deformation, images, defo_counters)]

    image_union = [ConcatLayer(
        incomings=[FlattenLayer(b), FlattenLayer(s), FlattenLayer(d)],
        name='union_%s' % i
    ) for b, s, d, i in zip(baseline, subtraction, deformation, images)]

    dense = [DenseLayer(
        incoming=u,
        name='\033[32mdense_%s\033[0m' % i,
        num_units=dense_size,
        nonlinearity=nonlinearities.softmax
    ) for u, i in zip(image_union, images)]

    union = ConcatLayer(
        incomings=dense,
        name='union'
    )

    soft = DenseLayer(
        incoming=union,
        name='\033[32mclass_out\033[0m',
        num_units=2,
        nonlinearity=nonlinearities.softmax
    )

    return soft


def get_convolutional_block(
        incoming,
        convo_size=3,
        num_filters=32,
        pool_size=2,
        drop=0.5,
        padding='valid',
        counter=itertools.count(),
        sufix=''
):
    index = counter.next()
    convolution = Conv3DLayer(
        incoming=incoming,
        name='\033[34mconv_%s%d\033[0m' % (sufix, index),
        num_filters=num_filters,
        filter_size=convo_size,
        pad=padding
    )
    normalisation = batch_norm(
        layer=convolution,
        name='norm_%s%d' % (sufix, index)
    )
    dropout = DropoutLayer(
        incoming=normalisation,
        name='drop_%s%d' % (sufix, index),
        p=drop
    )
    pool = Pool3DLayer(
        incoming=dropout,
        name='\033[31mavg_pool_%s%d\033[0m' % (sufix, index),
        pool_size=pool_size,
        mode='average_inc_pad'
    )

    return pool


def get_shared_convolutional_block(
            incoming1,
            incoming2,
            convo_size=3,
            num_filters=32,
            pool_size=2,
            drop=0.5,
            padding='valid',
            counter=itertools.count(),
            sufix=''
):

    index = counter.next()

    convolution1 = Conv3DLayer(
        incoming=incoming1,
        name='\033[34mconv_%s_1_%d\033[0m' % (sufix, index),
        num_filters=num_filters,
        filter_size=convo_size,
        pad=padding
    )
    convolution2 = Conv3DLayer(
        incoming=incoming2,
        name='\033[34mconv_%s_2_%d\033[0m' % (sufix, index),
        num_filters=num_filters,
        filter_size=convo_size,
        W=convolution1.W,
        b=convolution1.b,
        pad=padding
    )
    normalisation1 = batch_norm(
        layer=convolution1,
        name='norm_%s_1_%d' % (sufix, index)
    )
    normalisation2 = batch_norm(
        layer=convolution2,
        name='norm_%s_2_%d' % (sufix, index)
    )
    dropout1 = DropoutLayer(
        incoming=normalisation1,
        name='drop_%s_1_%d' % (sufix, index),
        p=drop
    )
    dropout2 = DropoutLayer(
        incoming=normalisation2,
        name='drop_%s_2_%d' % (sufix, index),
        p=drop
    )
    pool1 = Pool3DLayer(
        incoming=dropout1,
        name='\033[31mavg_pool_%s_1_%d\033[0m' % (sufix, index),
        pool_size=pool_size,
        mode='average_inc_pad'
    )
    pool2 = Pool3DLayer(
        incoming=dropout2,
        name='\033[31mavg_pool_%s_2_%d\033[0m' % (sufix, index),
        pool_size=pool_size,
        mode='average_inc_pad'
    )

    return pool1, pool2


def get_convolutional_block2d(
            incoming,
            convo_size=3,
            num_filters=32,
            pool_size=2,
            drop=0.5,
            padding='valid',
            counter=itertools.count(),
            sufix=''
):
        index = counter.next()

        convolution = Conv2DLayer(
            incoming=incoming,
            name='\033[34mconv_%s%d\033[0m' % (sufix, index),
            num_filters=num_filters,
            filter_size=convo_size,
            pad=padding
        )
        normalisation = batch_norm(
            layer=convolution,
            name='norm_%s%d' % (sufix, index)
        )
        dropout = DropoutLayer(
            incoming=normalisation,
            name='drop_%s%d' % (sufix, index),
            p=drop
        )
        pool = MaxPool2DLayer(
            incoming=dropout,
            name='\033[31mavg_pool_%s%d\033[0m' % (sufix, index),
            pool_size=pool_size,
            mode='average_inc_pad'
        )

        return pool


def get_shared_convolutional_block2d(
        incoming1,
        incoming2,
        convo_size=3,
        num_filters=32,
        pool_size=2,
        drop=0.5,
        padding='valid',
        counter=itertools.count(),
        sufix='',
        nonlinearity=nonlinearities.very_leaky_rectify
):
    index = counter.next()

    convolution1 = Conv2DLayer(
        incoming=incoming1,
        name='\033[34mconv_%s1_%d\033[0m' % (sufix, index),
        num_filters=num_filters,
        filter_size=convo_size,
        pad=padding,
        nonlinearity=nonlinearity
    )
    convolution2 = Conv2DLayer(
        incoming=incoming2,
        name='\033[34mconv_%s2_%d\033[0m' % (sufix, index),
        num_filters=num_filters,
        filter_size=convo_size,
        W=convolution1.W,
        b=convolution1.b,
        pad=padding,
        nonlinearity=nonlinearity
    )
    dropout1 = DropoutLayer(
        incoming=convolution1,
        name='drop_%s1_%d' % (sufix, index),
        p=drop
    )
    dropout2 = DropoutLayer(
        incoming=convolution2,
        name='drop_%s2_%d' % (sufix, index),
        p=drop
    )
    pool1 = MaxPool2DLayer(
        incoming=dropout1,
        name='\033[31mmax_pool_%s1_%d\033[0m' % (sufix, index),
        pool_size=pool_size
    )
    pool2 = MaxPool2DLayer(
        incoming=dropout2,
        name='\033[31mmax_pool_%s2_%d\033[0m' % (sufix, index),
        pool_size=pool_size
    )

    return pool1, pool2


def get_lnet(in1, in2, sufix):
    counter = itertools.count()
    vnet1_1, vnet1_2 = get_shared_convolutional_block2d(in1, in2, 5, 24, sufix=sufix, drop=0.25, counter=counter)
    vnet2_1, vnet2_2 = get_shared_convolutional_block2d(vnet1_1, vnet1_2, 3, 32, sufix=sufix, drop=0.25)
    index = counter.next()
    convolution1 = Conv2DLayer(
        incoming=vnet2_1,
        name='\033[34mconv_%s1_%d\033[0m' % (sufix, index),
        num_filters=48,
        filter_size=3,
        nonlinearity=nonlinearities.very_leaky_rectify
    )
    convolution2 = Conv2DLayer(
        incoming=vnet2_2,
        name='\033[34mconv_%s2_%d\033[0m' % (sufix, index),
        num_filters=48,
        filter_size=3,
        W=convolution1.W,
        b=convolution1.b,
        nonlinearity=nonlinearities.very_leaky_rectify
    )
    dropout1 = DropoutLayer(
        incoming=convolution1,
        name='drop_%s1_%d' % (sufix, index),
        p=0.25
    )
    dropout2 = DropoutLayer(
        incoming=convolution2,
        name='drop_%s2_%d' % (sufix, index),
        p=0.25
    )
    union = ConcatLayer(
        incomings=[dropout1, dropout2],
        name='concat_%s' % sufix
    )
    convolutionf = Conv2DLayer(
        incoming=union,
        name='\033[34mconv_%sf\033[0m' % sufix,
        num_filters=48,
        filter_size=1,
        nonlinearity=nonlinearities.very_leaky_rectify
    )
    dense = DenseLayer(
        incoming=convolutionf,
        name='\033[32mlnet_%s_out\033[0m' % sufix,
        num_units=16,
        nonlinearity=nonlinearities.very_leaky_rectify
    )

    return dense


def create_classifier_net(
        layers,
        patience,
        name,
        obj_f='xent',
        epochs=200
):

    objective_function = {
        'xent': objectives.categorical_crossentropy,
        'pdsc': objective_f.probabilistic_dsc_objective,
        'ldsc': objective_f.logarithmic_dsc_objective
    }

    return NeuralNet(

        layers=layers,

        regression=False,
        objective_loss_function=objective_function[obj_f],
        custom_scores=[
            ('prob dsc', objective_f.accuracy_dsc_probabilistic),
            ('dsc', objective_f.accuracy_dsc),
        ],

        # update=updates.adadelta,
        update=updates.adam,
        update_learning_rate=1e-4,

        on_epoch_finished=get_epoch_finished(name, patience),

        batch_iterator_train=BatchIterator(batch_size=512),

        verbose=11,
        max_epochs=epochs
    )


def create_segmentation_net(
        layers,
        patience,
        name,
        custom_scores=None,
        epochs=200
):
    return NeuralNet(

        layers=layers,

        regression=True,

        update=updates.adam,
        update_learning_rate=1e-3,

        on_epoch_finished=get_epoch_finished(name, patience),

        custom_scores=custom_scores,

        objective_loss_function=objectives.categorical_crossentropy,

        verbose=11,
        max_epochs=epochs
    )


def create_cnn3d_longitudinal(
        convo_blocks,
        input_shape,
        images,
        convo_size,
        pool_size,
        dense_size,
        number_filters,
        padding,
        drop,
        defo,
        patience,
        name,
        epochs
):
    layer_list = get_layers_longitudinal(
        convo_blocks=convo_blocks,
        input_shape=input_shape,
        images=images,
        convo_size=convo_size,
        pool_size=pool_size,
        dense_size=dense_size,
        number_filters=number_filters,
        padding=padding,
        drop=drop,
    ) if not defo else get_layers_longitudinal_deformation(
        convo_blocks=convo_blocks,
        input_shape=input_shape,
        d_off=defo,
        images=images,
        convo_size=convo_size,
        pool_size=pool_size,
        dense_size=dense_size,
        number_filters=number_filters,
        padding=padding,
        drop=drop,
    )

    return create_classifier_net(
        layer_list,
        patience,
        name,
        epochs=epochs
    )


def create_cnn_greenspan(
            input_channels,
            patience,
            name,
            epochs
):
        layer_list = get_layers_greenspan(input_channels)

        return create_classifier_net(
            layer_list,
            patience,
            name,
            epochs=epochs
        )
