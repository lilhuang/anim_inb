dataset_root = '/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_png_example'
flow_root = '/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_npz_2stream_flow_example'

img_size = (2048 ,1024)
img_resize = None
patch_size = 512
random_reverse = False

lr = 1e-3
lr_stream = 1e-3
lr_d = 1e-4
beta1 = 0.5
warp_weight = 1
gan_weight = 0.55

encoder_name = "resnet34_7" #resnet34_5, resnet34_6, or resnet34_7
encoder_weights = None #imagenet or None
encoder_depth = 7 #5, 6, or 7
decoder_channels = (1024, 512, 256, 128, 64, 32, 16) #add to/take away from the front
in_channels = 32 #2 if only input frames to generator; 6 if also input flows

stream_share = False
stream_encoder_name = "resnet34_5" #resnet34_5, resnet34_6, or resnet34_7
stream_encoder_weights = None #imagenet or None
stream_encoder_depth = 5
stream_decoder_channels = (256, 128, 64, 32, 16) #add to/take away from the front
stream_in_channels = 3 #2 if only input frames to generator; 6 if also input flows

model = 'UNet'

cur_epoch = 0
num_epochs = 50

num_workers = 0
batch_size =  32 #for 1 gpu
test_batch_size = 8 #for 1 gpu
# batch_size = 80 #for 4 gpu
# test_batch_size = 8 #for 4 gpu

name = "suzannes_example"

metrics_dir = 'outputs/'+name+'_METRICS_results_'
train_store_path = 'outputs/'+name+'_TRAIN_results_'
test_store_path = 'outputs/'+name+'_TEST_results_'

loss_img_path = ''+name+'_img.png'
checkpoint_latest_dir = '/fs/cfar-projects/anim_inb/checkpoints/latest_model_'+name
checkpoint_latest_file = 'EPOCH_'

# checkpoint_in = checkpoint_latest_dir+"/"+checkpoint_latest_file+"90.pth"
# checkpoint_in_1 = checkpoint_latest_dir+"/"+checkpoint_latest_file+"90_1.pth"
# checkpoint_in_2 = checkpoint_latest_dir+"/"+checkpoint_latest_file+"90_2.pth"
checkpoint_in = None
checkpoint_loss = None
