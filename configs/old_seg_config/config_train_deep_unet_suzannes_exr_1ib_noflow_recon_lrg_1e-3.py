trainset_root = '/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_png'
testset_root = '/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_png'
csv_root = '/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_csv'
flow_root = '/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_npz_flow'

dataset = "suzanne_exr" #either "suzanne_exr", "blender_cubes", "suzanne", or "SU"
test_size = (2048, 1024)
patch_size = 512
test_resize = None
random_reverse = False
dt = False
num_ib_frames = 1
flow_type = "gt" #tvl1, raft/dl name, or gt
small_dataset = False

lr = 1e-3
lr_d = 1e-4
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
batch_size = 80 #for 1 gpu, 1 patch discriminators
test_batch_size = 20 #for 1 gpu
# batch_size = 200 #for 4 gpus, 1 patch discriminator
# test_batch_size = 1 #for 4 gpus

overfit = False

metrics_dir = 'outputs/avi_suzannes_exr_'+str(num_ib_frames)+'ib_recon_lrg_1e-3_deep_unet_noflow_METRICS_results'
train_store_path = 'outputs/avi_suzannes_exr_'+str(num_ib_frames)+'ib_recon_lrg_1e-3_deep_unet_noflow_TRAIN_results'
test_store_path = 'outputs/avi_suzannes_exr_'+str(num_ib_frames)+'ib_recon_lrg_1e-3_deep_unet_noflow_TEST_results'

loss_img_path = 'progress_suzannes_exr_'+str(num_ib_frames)+'ib_recon_lrg_1e-3_deep_unet_noflow_img.png'
checkpoint_latest_dir = '/fs/cfar-projects/anim_inb/checkpoints/latest_model_suzannes_exr_'+str(num_ib_frames)+'ib_recon_lrg_1e-3_deep_unet_noflow_'
checkpoint_latest_file = 'latest_model_suzannes_exr_'+str(num_ib_frames)+'ib_recon_lrg_1e-3_deep_unet_noflow_EPOCH_'

checkpoint_in = None
checkpoint_loss = None
