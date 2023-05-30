trainset_root = '/fs/cfar-projects/anim_inb/datasets/vid_pngs_dog'
testset_root = '/fs/cfar-projects/anim_inb/datasets/vid_pngs_dog'
csv_root = '/fs/cfar-projects/anim_inb/datasets/vid_pngs_csvs_seq'
csv_filename = '_JamesBaxterThumper_2s_1s_1ib.csv'
trainflow_root = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_vid_pngs_dog"
testflow_root = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_vid_pngs_dog"

dataset = "pt_JamesBaxterThumper" #either "suzanne_exr", "blender_cubes", "suzanne", or "SU"
test_size = (2048, 1024)
test_resize = (2048, 1024)
patch_size = 512
random_reverse = False
dt = False
num_ib_frames = 3
flow_type = "pips" #either "tvl1", "gt", or "raft" maybe but we're never using raft lol
small_dataset = False
csv = True

lr = 1e-3
lr_stream = 1e-4
lr_d = 1e-4
beta1 = 0.5
warp_weight = 1
gan_weight = 0.55

encoder_name = "resnet34_7" #resnet34_5, resnet34_6, or resnet34_7
encoder_weights = None #imagenet or None
encoder_depth = 7 #5, 6, or 7
decoder_channels = (1024, 512, 256, 128, 64, 32, 16) #add to/take away from the front
in_channels = 32 #2 if only input frames to generator; 6 if also input flows

stream_encoder_name = "resnet34_5" #resnet34_5, resnet34_6, or resnet34_7
stream_encoder_weights = None #imagenet or None
stream_encoder_depth = 5 #5, 6, or 7
stream_decoder_channels = (256, 128, 64, 32, 16) #add to/take away from the front
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
num_epochs = 201

num_workers = 4
batch_size = 4 #for 1 gpu, 1 patch discriminators
test_batch_size = 4 #for 1 gpu
# batch_size = 28 #for 4 gpus, 1 patch discriminator
# test_batch_size = 8 #for 4 gpus

overfit = False

namename = "final_pt_JamesBaxterThumper_"

metrics_dir = 'outputs/'+namename+str(num_ib_frames)+'ib_seq_METRICS_results'
train_store_path = 'outputs/'+namename+str(num_ib_frames)+'ib_seq_TRAIN_results'
test_store_path = 'outputs/'+namename+str(num_ib_frames)+'ib_seq_TEST_results'

loss_img_path = 'progress'+namename+str(num_ib_frames)+'_seq_img.png'
checkpoint_latest_dir = '/fs/cfar-projects/anim_inb/checkpoints/latest_model'+namename+str(num_ib_frames)+'ib_seq'
checkpoint_latest_file = 'latest_model'+namename+str(num_ib_frames)+'ib_seq_EPOCH_'

# checkpoint_in = checkpoint_latest_dir + "/" + checkpoint_latest_file + "230.pth"
checkpoint_in = None
checkpoint_loss = None





