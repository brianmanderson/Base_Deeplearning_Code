import os, sys
from Data_Generators.Generators import Train_Data_Generator3D, Image_Clipping_and_Padding
from keras.models import *
import tensorflow as tf
from keras.optimizers import Adam
import pandas as pd
from Keras_Utils.Keras_Utilities import dice_coef_3D_np, ModelCheckpoint_new, get_available_gpus, save_obj,load_obj, \
    remove_non_liver, weighted_categorical_crossentropy, weighted_categorical_crossentropy_masked, dice_coef_3D, np, Fill_Missing_Segments
from Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Callbacks.Visualizing_Model_Utils import TensorBoardImage
from Models.Keras_3D_Models import my_3D_UNet
from Cyclical_Learning_Rate.clr_callback import CyclicLR
from Data_Generators.Return_Paths import Path_Return_Class


try:
    base = r'\\mymdafiles\di_data1'
    base_path = os.path.join(base,r'Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Segments\New_Niftii_Arrays\CT')
    os.listdir(base_path)
except:
    base_path = os.path.join('..', '..', '..', '..','..', '..', 'Liver_Segments')
    base = os.path.join('..', '..', '..', '..','..', '..')
morfeus_drive = os.path.abspath(os.path.join(base,'Morfeus','BMAnderson','CNN','Data','Data_Liver','Liver_Segments'))
paths = [os.path.join(base_path, 'Train', 'Single_Images3D')]
paths_test_generator = [os.path.join(base_path, 'Test', 'Single_Images3D')]
paths_validation_generator = [os.path.join(base_path, 'Validation', 'Single_Images3D')]

def run_model(gpu=1,min_lr=1e-4, max_lr=1e-2, layers_dict=None, epochs=1000, model_name = '3D_Atrous',validation_generator=None,
              step_size_factor=5,add='', train_generator=None, test_generator=None,batch_norm=False,return_mask=False, **kwargs):
    G = get_available_gpus()
    if len(G) == 1:
        gpu = 0
    with tf.device('/gpu:' + str(gpu)):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        K.set_session(sess)
        if not os.path.exists(morfeus_drive):
            print('Morf wrong')
            return None
        if not os.path.exists(base_path):
            print('base wrong')
            return None

        x,y = train_generator.__getitem__(0)
        do_validation = False
        if do_validation:
            print('Doing validation')
            model_path = r'K:\Morfeus\Auto_Contour_Sites\Models\Liver_Segments\weights-improvement-best.hdf5'
            Model_val = load_model(model_path,
                                   custom_objects={'loss':weighted_categorical_crossentropy(
                                       np.load(os.path.join('.','class_weights.npy'))),
                                       'dice_coef_3D': dice_coef_3D})
            data_dict = {}
            Fill_Missing_Segments_Class = Fill_Missing_Segments()
            for i in range(len(validation_generator)):
                print(i)
                data_dict[i] = {}
                x,y = validation_generator.__getitem__(i)
                pred = Model_val.predict(x)[0,...]
                x = x[0,...]
                mask = x[...,0] != -3.55
                pred[mask == 0] = 0
                for ii in range(1, pred.shape[-1]):
                    pred[..., ii] = remove_non_liver(pred[..., ii], do_2D=True)
                amounts = np.sum(pred, axis=(1, 2))
                indexes = np.where(
                    (np.max(amounts[:, (5, 6)], axis=-1) > 0) & (np.max(amounts[:, (7, 8)], axis=-1) > 0))
                if indexes:
                    indexes = indexes[0]
                    for ii in indexes:
                        if amounts[ii, 5] < amounts[ii, 8]:
                            pred[ii, ..., 5] = 0
                        else:
                            pred[ii, ..., 8] = 0
                        if amounts[ii, 6] < amounts[i, 7]:
                            pred[ii, ..., 6] = 0
                        else:
                            pred[ii, ..., 7] = 0
                pred = Fill_Missing_Segments_Class.make_distance_map(pred,mask,spacing=[1,1,100])
                for ii in range(1,9):
                    dsc = dice_coef_3D_np(y[0,...,ii],pred[...,ii])
                    print(dsc)
                    data_dict[i][ii] = dsc
            # xxx = 1
            # save_obj(os.path.join('.','out_dsc.pkl'), data_dict)
            df = pd.DataFrame(data_dict).T  # transpose to look just like the sheet above
            df.to_excel(os.path.join('.', 'no_processing.xlsx'))
            return None
        paths_class = Path_Return_Class(base_path=base_path, morfeus_path=morfeus_drive)
        num_cycles = int(epochs / (2 * step_size_factor))
                                                                    #  vv Change this
        things = ['{}_Layers'.format(kwargs['Layers']),'{}_Filters_BatchNorm_maskedpred'.format(kwargs['filters']),
                        '{}_MinLR_{}_MaxLR'.format(min_lr,max_lr),'{}_step_size_{}_cycles'.format(step_size_factor,num_cycles)]
        paths_class.define_model_things(model_name,things,add)

        model_path_out = paths_class.model_path_out
        tensorboard_output = paths_class.tensorboard_path_out

        epoch_i = 0
        optimizer = Adam(lr=min_lr)
        period = 5
        checkpoint = ModelCheckpoint_new(model_path_out, monitor='val_dice_coef_3D', verbose=1, save_best_only=True,
                                         save_weights_only=False, period=period, mode='max')

        tensorboard = TensorBoardImage(log_dir=tensorboard_output, batch_size=1, write_graph=True, write_grads=False,num_images=3,
                                       update_freq='epoch',  data_generator=test_generator, image_frequency=period)
        lrate = CyclicLR(base_lr=min_lr, max_lr=max_lr, step_size=len(train_generator) * step_size_factor, mode='triangular2', pre_cycle=1)
        callbacks = [lrate, checkpoint, tensorboard]
        loss = weighted_categorical_crossentropy(np.load(os.path.join('.','new_class_weights.npy'))) #categorical_crossentropy
        # else:
        #     loss = weighted_categorical_crossentropy_masked(np.load(os.path.join('.','new_class_weights.npy')))
        model = my_3D_UNet(filter_vals=(3, 3, 3), layers_dict=layers_dict, pool_size=(2, 2, 2),custom_loss=loss,
                               activation='elu', pool_type='Max',out_classes=9, mask_loss=False,mask_output=True)
        # if return_mask:
        #     loss = weighted_categorical_crossentropy(np.load(os.path.join('.', 'new_class_weights.npy')))
        Model_val = model.created_model
        train = True
        if train:
            Model_val.compile(optimizer, loss=loss, metrics=['accuracy', dice_coef_3D])
            Model_val.fit_generator(generator=train_generator, workers=10, use_multiprocessing=False, max_queue_size=50,
                                    shuffle=True, epochs=epochs, callbacks=callbacks, initial_epoch=epoch_i,
                                    validation_data=test_generator,steps_per_epoch=len(train_generator))


def get_layers_dict(layers=1, filters=16, conv_blocks=1, num_conv_blocks=None, num_atrous_blocks=4, max_blocks=2, **kwargs):
    atrous_rate = 2
    layers_dict = {}
    pool = (4, 4, 4)
    for layer in range(conv_blocks):
        conv_block = {'Channels': [filters]}
        if num_conv_blocks is not None:
            conv_blocks_total = [conv_block for _ in range(num_conv_blocks)]
        else:
            conv_blocks_total = [conv_block]
        layers_dict['Layer_' + str(layer)] = {'Encoding':conv_blocks_total,'Pooling':pool,'Decoding':conv_blocks_total}
        pool = (2, 2, 2)
        filters = int(filters*2)
    pool = (2, 2, 2)
    for layer in range(conv_blocks,layers-1):
        atrous_block = {'Channels': [filters], 'Atrous_block': [atrous_rate], 'Kernel': [(3, 3, 3)]}
        layers_dict['Layer_' + str(layer)] = {'Encoding': [atrous_block for _ in range(num_atrous_blocks)], 'Pooling': pool,
                                              'Decoding': [atrous_block for _ in range(num_atrous_blocks)]}
        filters = int(filters*2)
        num_atrous_blocks = min([(num_atrous_blocks) * 2,max_blocks])
    num_atrous_blocks = min([(num_atrous_blocks) * 2, max_blocks])
    atrous_block = {'Channels': [filters], 'Atrous_block': [atrous_rate],'Kernel': [(3, 3, 3)]}
    layers_dict['Base'] = {'Encoding':[atrous_block for _ in range(num_atrous_blocks)]}
    return layers_dict
def get_layers_dict_conv(layers, filters, conv_blocks, num_conv_blocks=None, **kwargs):
    layers_dict = {}
    pool = (4, 4, 4)
    for layer in range(layers-1):
        conv_block = {'Channels': [filters]}
        if num_conv_blocks is not None:
            conv_blocks_total = [conv_block for _ in range(num_conv_blocks)]
        else:
            conv_blocks_total = [conv_block]
        layers_dict['Layer_' + str(layer)] = {'Encoding':conv_blocks_total,'Pooling':pool,'Decoding':conv_blocks_total}
        pool = (2, 2, 2)
        filters = int(filters*2)
    conv_block = {'Channels': [filters]}
    layers_dict['Base'] = {'Encoding':[conv_block for _ in range(num_conv_blocks)]}
    return layers_dict

if __name__ == '__main__':
    k = np.load(os.path.join('.', 'class_weights.npy'))
    # k[0] = 0.0
    k[1] = 10.0
    k[2] = 5.0
    np.save(os.path.join('.', 'new_class_weights.npy'),k)
    loss = weighted_categorical_crossentropy(
        np.load(os.path.join('.', 'new_class_weights.npy')))  # categorical_crossentropy
    perturbations = None
    num_classes = 9
    batch_size = 1
    mean_val = 97
    std_val = 53
    image_num = 1
    desired_output = None
    expansion = 5
    z_images = 64
    clip = [0, 0, 0]
    train_generator = Train_Data_Generator3D(batch_size=image_num,
                                             whole_patient=True, shuffle=False,
                                              num_patients=batch_size,z_images=z_images,
                                             data_paths=paths, num_classes=num_classes,clip=clip,perturbations=perturbations,
                                             flatten=False, mean_val=mean_val, std_val=std_val, expansion=expansion, three_layer=False,
                                             is_test_set=True,all_for_one=False,verbose=False,output_size=desired_output,
                                             noise=0.05, write_predictions=False)
    test_generator = Train_Data_Generator3D(batch_size=image_num,whole_patient=True, shuffle=False,
                                            num_patients=batch_size,z_images=z_images,data_paths=paths_test_generator,
                                            num_classes=num_classes,clip=clip,perturbations=perturbations,
                                            flatten=False, mean_val=mean_val, std_val=std_val, expansion=expansion,
                                            three_layer=False,is_test_set=True,all_for_one=False,verbose=False,
                                            output_size=desired_output,noise=0.05, write_predictions=False)
    if os.path.exists(paths_validation_generator[0]):
        validation_generator = Train_Data_Generator3D(batch_size=image_num,whole_patient=True, shuffle=False,
                                                num_patients=batch_size,z_images=z_images,data_paths=paths_validation_generator,
                                                num_classes=num_classes,clip=clip,perturbations=perturbations,
                                                flatten=False, mean_val=mean_val, std_val=std_val, expansion=expansion,
                                                three_layer=False,is_test_set=True,all_for_one=False,verbose=False,
                                                output_size=desired_output,noise=0.05, write_predictions=False)

    gpu = 1
    step_size_factor = 8
    num_cycles = 12
    base_things = {'num_conv_blocks':2,'conv_blocks':1,'num_convs':2,'num_atrous_blocks':1,'step_size_factor':step_size_factor,'num_cycles':num_cycles}
    base_things_conv = {'num_conv_blocks':2,'conv_blocks':1,'num_convs':2,'num_atrous_blocks':0,'step_size_factor':step_size_factor,'num_cycles':num_cycles}
    base_dict = lambda min_lr,max_lr,filters: {'min_lr':min_lr,'max_lr':max_lr,'filters':filters}
    data_dict = {
        3: [
            {'min_lr': 2e-6, 'max_lr': 3e-3, 'filters': 8},
            {'min_lr': 8.4e-7, 'max_lr': 2e-3, 'filters': 16},
            {'min_lr': 3e-7, 'max_lr': 9e-4, 'filters': 32}

        ],
        4: [
            {'min_lr': 7.6e-7, 'max_lr': 2e-3, 'filters': 8},
            {'min_lr': 2.5e-7, 'max_lr': 1.9e-3, 'filters': 16},
            {'min_lr': 1e-7, 'max_lr': 1e-3, 'filters': 32}
        ],
        5: [
            {'min_lr': 8e-7, 'max_lr': 1e-3, 'filters': 8},
            {'min_lr': 1.5e-7, 'max_lr': 8e-4, 'filters': 16},
            {'min_lr': 7e-8, 'max_lr': 1e-3, 'filters': 32}
        ]
    }
    data_dict_4pool = {
        3: [
            {'min_lr': 9e-7, 'max_lr': 8e-3, 'filters': 8},
            {'min_lr': 8.4e-7, 'max_lr': 2e-3, 'filters': 16},
            {'min_lr': 5e-7, 'max_lr': 4e-4, 'filters': 32}

        ],
        4: [
            {'min_lr': 8e-7, 'max_lr': 6e-3, 'filters': 8},
            {'min_lr': 2.5e-7, 'max_lr': 1.9e-3, 'filters': 16},
            {'min_lr': 1.5e-7, 'max_lr': 3e-3, 'filters': 32}
        ],
        5: [
            {'min_lr': 7e-7, 'max_lr': 9e-3, 'filters': 8}
        ]
    }
    data_dict_weighted = {
        3: [
            #base_dict(1e-6,1.7e-3,8),
            #base_dict(3e-6, 1e-3, 16),
            base_dict(1e-6, 5e-4, 32)
        ],
        4: [
            #base_dict(1e-6, 8e-4, 8),
            #base_dict(1e-6, 6e-4, 16),
            base_dict(5e-7, 5e-4, 32)
        ]
    }
    data_dict_weighted_0 = {
        3: [
            base_dict(5e-6,1.2e-3,16),
            base_dict(1e-6, 3e-4, 32)
            # base_dict(1e-5, 2e-3, 8)
        ],
        4: [
            # base_dict(2e-6, 1e-3, 8)
            base_dict(1e-6, 8e-4, 16),
            base_dict(1e-7, 2e-4, 32)
        ],
        5: [
            # base_dict(3e-6, 8e-4, 8)
            base_dict(1e-6, 4e-4, 16),
            base_dict(2e-7, 2e-4, 32)
        ]
    }
    data_dict_weighted_10_mask = {
        3: [
            base_dict(2e-6,6e-4,16),
            base_dict(1e-6, 3e-4, 32)
            # base_dict(1e-5, 2e-3, 8)
        ],
        4: [
            # base_dict(2e-6, 1e-3, 8)
            base_dict(1e-6, 8.5e-4, 16),
            base_dict(1e-6, 2.3e-4, 32)
        ],
        5: [
            # base_dict(3e-6, 8e-4, 8)
            base_dict(1e-6, 5e-4, 16),
            base_dict(1e-6, 2e-4, 32)
        ]
    }
    data_dict_weighted_10_mask_0atrous = {
        3: [
            base_dict(1e-6,6.9e-4,16),
            base_dict(1e-6, 3e-4, 32)
        ],
        4: [
            base_dict(8e-6, 2.7e-4, 16),
            base_dict(6e-6, 1.9e-4, 32)
        ],
        5: [
            base_dict(5e-6, 2.8e-4, 16),
            base_dict(5e-6, 1.3e-4, 32)
        ]
    }
    data_dict_weighted_10_mask_0atrous_BN = {
        3: [
            base_dict(2e-6,2e-3,16),
            base_dict(2e-7, 1.3e-3, 32)
        ],
        4: [
            base_dict(8e-7, 8e-4, 16),
            base_dict(3e-7, 5e-4, 32)
        ],
        5: [
            base_dict(2e-7, 3e-4, 16),
            base_dict(3e-7, 1e-4, 32)
        ]
    }
    data_dict_weighted_10_mask_0atrous_BN_curated = {
        3: [
            base_dict(5e-7, 1.3e-3, 32)
        ],
        4: [
            base_dict(5e-7, 5e-4, 32)
        ]
    }
    data_dict_weighted_10_mask_0atrous_BN_curated_mask_pred = {
        3: [
            base_dict(1.3e-6, 1.4e-3, 32),
            base_dict(6e-6, 2e-3, 16)
        ],
        4: [
            base_dict(5e-7, 3e-4, 32),
            base_dict(2e-6, 1.4e-3, 16)
        ]
    }
    data_dict_curated_weighted = {
        3: [
            base_dict(1.3e-6, 1.6e-3, 32),
        ]
    }
    mask_output = True
    return_mask = True
    batch_norm = True
    model_name = '3D_Atrous_curated' # change this
    for iteration in range(3):
        for layer in [3]:
            data = data_dict_curated_weighted[layer] # change this
            for run_data in data:
                run_data.update(base_things) # Change this
                run_data['Layers'] = str(layer)
                layers_dict = get_layers_dict(layers=layer, **run_data)
                # layers_dict = get_layers_dict_conv(layers=layer, **run_data) # change this
                train_generator_3D = Image_Clipping_and_Padding(layers_dict, train_generator, return_mask=return_mask,
                                                                liver_box=True,mask_output=mask_output)
                test_generator_3D = Image_Clipping_and_Padding(layers_dict, test_generator, return_mask=return_mask,
                                                               liver_box=True,mask_output=mask_output)
                # x,y = train_generator_3D.__getitem__(0)
                validation_generator_3D = None
                if os.path.exists(paths_validation_generator[0]):
                    validation_generator_3D = Image_Clipping_and_Padding(layers_dict, validation_generator,
                                                                         return_mask=False,liver_box=True,
                                                                         mask_output=mask_output)
                try:
                    run_model(gpu=gpu,layers_dict=layers_dict,train_generator=train_generator_3D,
                              test_generator=test_generator_3D,batch_norm=batch_norm,validation_generator=validation_generator_3D,
                              add='Iteration_' + str(iteration),return_mask=return_mask,
                              epochs=step_size_factor * 2 * num_cycles,model_name=model_name,**run_data)
                    K.clear_session()
                except:
                    K.clear_session()
