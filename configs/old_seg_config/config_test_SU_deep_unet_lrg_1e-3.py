testset_root = '/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/'
csv_root = '/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_csv'
flow_root = '/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_tvl1_flows'

dataset = "SU" #either "suzanne_exr", "blender_cubes", "suzanne", or "SU"
test_size = (2048, 1024)
patch_size = 512
test_resize = None
random_reverse = False
dt = False
num_ib_frames = 1
flow_type = "tvl1" #tvl1, raft/dl name, or gt
small_dataset = False

lr = 1e-3
lr_d = 1e-5
beta1 = 0.5
# warp_weight = 0.1
warp_weight = 1

# model = 'UNet_RRDB' #either UNet or UNet_RRDB
model = 'UNet'
discrim = None #either 'patch', 'multiple patch', or None
mask_loss = True

recon_loss = True
gan_loss = False
warp_loss = False

cur_epoch = 0
num_epochs = 61

num_workers = 4
# batch_size = 80 #for 1 gpu, 1 patch discriminators
# test_batch_size = 20 #for 1 gpu
batch_size = 200 #for 4 gpus, 1 patch discriminator
test_batch_size = 1 #for 4 gpus

overfit = False

dataset_root_filepath_test = "/fs/cfar-projects/anim_inb/datasets/pickles/test_SU_"+str(num_ib_frames)+"ib_"+str(test_size[0])+"x"+str(test_size[1])

metrics_dir = 'outputs/avi_SU_test_'+str(num_ib_frames)+'ib_recon_lrg_1e-3_deep_unet_METRICS_results'
test_store_path = 'outputs/avi_SU_'+str(num_ib_frames)+'ib_recon_lrg_1e-3_deep_unet_TEST_results'

checkpoint_latest_dir = '/fs/cfar-projects/anim_inb/checkpoints/latest_model_suzannes_exr_'+str(num_ib_frames)+'ib_recon_lrg_1e-3_lrd_1e-5_deep_unet_'
checkpoint_latest_file = 'latest_model_suzannes_exr_'+str(num_ib_frames)+'ib_recon_lrg_1e-3_lrd_1e-5_deep_unet_EPOCH_'

checkpoint_in = checkpoint_latest_dir + "/" + checkpoint_latest_file + "60.pth"
checkpoint_loss = None
