trainset_root = '/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog'
testset_root = '/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog'
csv_root = '/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_640x360_smol_csv'
trainflow_root = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_640x360_3ib"
testflow_root = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_640x360_3ib"

dataset = "SU" #either "blender_cubes", "suzanne", or "SU"
test_size = (640, 360)
test_resize = (640, 360)
random_reverse = False
dt = False
num_ib_frames = 3
flow_type = "pips" #pips, tvl1, gt, or None
small_dataset = False
csv = True

lr = 1e-3
lr_d = 1e-4
beta1 = 0.5
warp_weight = 1
gan_weight = 0.55

encoder_name = "resnet34_7" #resnet34_5, resnet34_6, or resnet34_7
encoder_weights = None #imagenet or None
encoder_depth = 7 #5, 6, or 7
decoder_channels = (1024, 512, 256, 128, 64, 32, 16) #add to/take away from the front
in_channels = 4 #2 if only input frames to generator; 6 if also input flows

# model = 'UNet_RRDB' #either UNet or UNet_RRDB
model = 'UNet'
discrim = None #either 'patch', 'multiple patch', or None
mask_loss = True
l1_loss = False
l2_loss = False

recon_loss = True
gan_loss = True
warp_loss = False

cur_epoch = 0
num_epochs = 101

num_workers = 4
batch_size = 20 #for 1 gpu, 1 patch discriminators
test_batch_size = 20 #for 1 gpu
# batch_size = 28 #for 4 gpus, 1 patch discriminator
# test_batch_size = 8 #for 4 gpus

overfit = False

dataset_root_filepath_train = "/fs/cfar-projects/anim_inb/datasets/pickles/train_SU_"+str(num_ib_frames)+"ib_"+str(test_size[0])+"x"+str(test_size[1])+"_smol"
dataset_root_filepath_test = "/fs/cfar-projects/anim_inb/datasets/pickles/test_SU_"+str(num_ib_frames)+"ib_"+str(test_size[0])+"x"+str(test_size[1])+"_smol"

metrics_dir = 'outputs/avi_SU_smol_warp_then_train_'+str(num_ib_frames)+'ib_lrg_1e-3_lrd_1e-4_deep7_unet_recon_gan_small_METRICS_results'
train_store_path = 'outputs/avi_SU_smol_warp_then_train_'+str(num_ib_frames)+'ib_lrg_1e-3_lrd_1e-4_deep7_unet_recon_gan_small_TRAIN_results'
test_store_path = 'outputs/avi_SU_smol_warp_then_train_'+str(num_ib_frames)+'ib_lrg_1e-3_lrd_1e-4_deep7_unet_recon_gan_small_TEST_results'

loss_img_path = 'progress_SU_smol_warp_then_train_'+str(num_ib_frames)+'ib_lrg_1e-3_lrd_1e-4_deep7_unet_recon_gan_small_img.png'
checkpoint_latest_dir = '/fs/cfar-projects/anim_inb/checkpoints/latest_model_SU_smol_warp_then_train_'+str(num_ib_frames)+'ib_lrg_1e-3_lrd_1e-4_deep7_unet_recon_gan_small'
checkpoint_latest_file = 'latest_model_SU_smol_warp_then_train_'+str(num_ib_frames)+'ib_lrg_1e-3_lrd_1e-4_deep7_unet_recon_gan_small_EPOCH_'

# checkpoint_in = checkpoint_latest_dir + "/" + checkpoint_latest_file + "16.pth"
checkpoint_in = None
checkpoint_loss = None
