trainset_root = '/fs/cfar-projects/anim_inb/new_datasets/moving_gif_png_dog/train'
testset_root = '/fs/cfar-projects/anim_inb/new_datasets/moving_gif_png_dog/test'
csv_root = '/fs/cfar-projects/anim_inb/new_datasets/moving_gif_csvs'
trainflow_root = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_moving_gif_dog/train"
testflow_root = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_moving_gif_dog/test"

# dataset = "SU" #either "blender_cubes", "suzanne", or "SU"
# dataset = "all"
dataset = "moving_gif"
test_size = (360, 360)
test_resize = (360, 360)
random_reverse = False
dt = False
num_ib_frames = 1
flow_type = "pips" #pips, tvl1, gt, or None
small_dataset = False
csv = True

lr = 1e-3
lr_stream = 1e-4
lr_d = 1e-4
beta1 = 0.5
warp_weight = 1
gan_weight = 0.55

encoder_name = "resnet34_5" #resnet34_5, resnet34_6, or resnet34_7
encoder_weights = None #imagenet or None
encoder_depth = 5 #5, 6, or 7
decoder_channels = (256, 128, 64, 32, 16) #add to/take away from the front
in_channels = 32 #2 if only input frames to generator; 6 if also input flows

stream_share = True
stream_encoder_name = "resnet34_5" #resnet34_5, resnet34_6, or resnet34_7
stream_encoder_weights = None #imagenet or None
stream_encoder_depth = 4
stream_decoder_channels = (128, 64, 32, 16) #add to/take away from the front
stream_in_channels = 3 #2 if only input frames to generator; 6 if also input flows

# model = 'UNet_RRDB' #either UNet or UNet_RRDB
model = 'UNet'
discrim = None #either 'patch', 'multiple patch', or None
mask_loss = True
l1_loss = False
l2_loss = False

recon_loss = True
gan_loss = False
warp_loss = False

cur_epoch = 0
num_epochs = 100

num_workers = 0
# batch_size = 4 #for 2 gpu
# test_batch_size = 8 #for 2 gpu
batch_size = 2 #for 1 gpu
test_batch_size = 1 #for 1 gpu
# batch_size = 28 #for 4 gpus, 1 patch discriminator
# test_batch_size = 8 #for 4 gpus

overfit = False

name = "moving_gif"

metrics_dir = 'outputs/'+name+'_METRICS_results_'
train_store_path = 'outputs/'+name+'_TRAIN_results_'
test_store_path = 'outputs/'+name+'_TEST_results_'

loss_img_path = name+'_img.png'
checkpoint_latest_dir = '/fs/cfar-projects/anim_inb/checkpoints/latest_model_'+name
checkpoint_latest_file = 'EPOCH_'

# checkpoint_in = checkpoint_latest_dir+"/"+checkpoint_latest_file+"90.pth"
# checkpoint_in_1 = checkpoint_latest_dir+"/"+checkpoint_latest_file+"90_1.pth"
# checkpoint_in_2 = checkpoint_latest_dir+"/"+checkpoint_latest_file+"90_2.pth"
checkpoint_in = None
checkpoint_loss = None
