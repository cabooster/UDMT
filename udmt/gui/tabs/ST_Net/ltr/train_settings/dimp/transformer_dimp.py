import torch.optim as optim
# from ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq
# from ltr.data import processing, sampler, LTRLoader
# from ltr.models.tracking import dimpnet
# import ltr.models.loss as ltr_losses
# import ltr.models.loss.kl_regression as klreg_losses
# import ltr.actors.tracking as tracking_actors
# from ltr.trainers import LTRTrainer
# import ltr.data.transforms as tfm
# from ltr import MultiGPU
# from udmt.gui.tabs.ST_Net.ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq
from udmt.gui.tabs.ST_Net.ltr.dataset import Got10k
from udmt.gui.tabs.ST_Net.ltr.data import processing, sampler, LTRLoader
from udmt.gui.tabs.ST_Net.ltr.models.tracking import dimpnet
from udmt.gui.tabs.ST_Net.ltr.models import loss as ltr_losses
from udmt.gui.tabs.ST_Net.ltr.models.loss import kl_regression as klreg_losses
from udmt.gui.tabs.ST_Net.ltr.actors import tracking as tracking_actors
from udmt.gui.tabs.ST_Net.ltr.trainers import LTRTrainer
from udmt.gui.tabs.ST_Net.ltr.data import transforms as tfm
from udmt.gui.tabs.ST_Net.ltr import MultiGPU
from udmt.gui import BASE_DIR

def run(settings,run_training_params):
    settings.description = 'Transformer-assisted tracker. Our baseline approach is SuperDiMP'
    settings.batch_size = run_training_params['batch_size']
    settings.num_workers = run_training_params['num_workers']
    settings.multi_gpu = True
    settings.print_interval = 10
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 3 #!!! white 3
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 4
    settings.feature_sz = 22 #22
    settings.output_sz = settings.feature_sz * 16 #  * 16
    settings.center_jitter_factor = {'train': 1, 'test': 1} #!! 2 2
    settings.scale_jitter_factor = {'train': 0, 'test': 0}
    # settings.center_jitter_factor = {'train': 3, 'test': 5.5}
    # settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}
    settings.hinge_threshold = 0.05
    # settings.print_stats = ['Loss/total', 'Loss/iou', 'ClfTrain/init_loss', 'ClfTrain/test_loss']

    # Train datasets
    # lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')#!!!!!!!
    # trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
    # coco_train = MSCOCOSeq(settings.env.coco_dir)

    # Validation datasets
    # got10k_val = Got10k(settings.env.got10k_dir, split='votval')


    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip(probability=0.5),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))
    # transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
    #                               tfm.RandomHorizontalFlip(probability=0.5))
    # transform_val = tfm.Transform(tfm.ToTensor(),
    #                               tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # The tracking pairs processing module
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    settings.sigma_factor_label = output_sigma
    proposal_params = {'boxes_per_frame': 128, 'gt_sigma': (0.05, 0.05), 'proposal_sigma': [(0.05, 0.05), (0.5, 0.5)]}
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}
    label_density_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}

    data_processing_train = processing.KLDiMPProcessing(search_area_factor=settings.search_area_factor,
                                                        output_sz=settings.output_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        crop_type='inside_major',
                                                        max_scale_change=1.5,
                                                        mode='sequence',
                                                        proposal_params=proposal_params,
                                                        label_function_params=label_params,
                                                        label_density_params=label_density_params,
                                                        transform=transform_train,
                                                        joint_transform=transform_joint)

    # data_processing_val = processing.KLDiMPProcessing(search_area_factor=settings.search_area_factor,
    #                                                   output_sz=settings.output_sz,
    #                                                   center_jitter_factor=settings.center_jitter_factor,
    #                                                   scale_jitter_factor=settings.scale_jitter_factor,
    #                                                   crop_type='inside_major',
    #                                                   max_scale_change=1.5,
    #                                                   mode='sequence',
    #                                                   proposal_params=proposal_params,
    #                                                   label_function_params=label_params,
    #                                                   label_density_params=label_density_params,
    #                                                   transform=transform_val,
    #                                                   joint_transform=transform_joint)

    # Train sampler and loader
    # dataset_train = sampler.DiMPSampler([lasot_train, got10k_train, trackingnet_train, coco_train], [1,1,1,1],
    #                                     samples_per_epoch=50000, max_gap=500, num_test_frames=3, num_train_frames=3,
    #                                     processing=data_processing_train)
    dataset_train = sampler.DiMPSampler([got10k_train], [1],
                                        samples_per_epoch=5000, max_gap=10, num_test_frames=3, num_train_frames=3,
                                        processing=data_processing_train,train_num_per_sequence = 2000) # !!5000

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    # Validation samplers and loaders
    # dataset_val = sampler.DiMPSampler([got10k_val], [1], samples_per_epoch=10000, max_gap=500,
    #                                   num_test_frames=3, num_train_frames=3,
    #                                   processing=data_processing_val)

    # loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
    #                        shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)

    # Create network and actor
    net = dimpnet.dimpnet50(filter_size=settings.target_filter_sz, backbone_pretrained=True, optim_iter=5,
                            clf_feat_norm=True, clf_feat_blocks=0, final_conv=True, out_feature_dim=512,
                            optim_init_step=0.9, optim_init_reg=0.1,
                            init_gauss_sigma=output_sigma * settings.feature_sz, num_dist_bins=100,
                            bin_displacement=0.1, mask_init_factor=3.0, target_mask_act='sigmoid', score_act='relu',
                            frozen_backbone_layers=['conv1', 'bn1', 'layer1', 'layer2'])

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    objective = {'bb_ce': klreg_losses.KLRegression(), 'test_clf': ltr_losses.LBHinge(threshold=settings.hinge_threshold)}

    # loss_weight = {'bb_ce': 0.01, 'test_clf': 100, 'test_init_clf': 100, 'test_iter_clf': 400}
    loss_weight = {'bb_ce': 0, 'test_clf': 100, 'test_init_clf': 100, 'test_iter_clf': 400}

    actor = tracking_actors.KLDiMPActor(net=net, objective=objective, loss_weight=loss_weight)

    # Optimizer
    
    optimizer = optim.Adam([{'params': actor.net.classifier.filter_initializer.parameters(), 'lr': 5e-5},
                            {'params': actor.net.classifier.filter_optimizer.parameters(), 'lr': 5e-4},
                            {'params': actor.net.classifier.feature_extractor.parameters(), 'lr': 5e-5},
                            {'params': actor.net.classifier.transformer.parameters(), 'lr': 1e-3},
                            {'params': actor.net.bb_regressor.parameters(), 'lr': 1e-3},
                            {'params': actor.net.feature_extractor.layer3.parameters(), 'lr': 2e-5}],
                           lr=2e-4)
    '''
    optimizer = optim.Adam([{'params': actor.net.classifier.filter_initializer.parameters(), 'lr': 5e-6},
                            {'params': actor.net.classifier.filter_optimizer.parameters(), 'lr': 5e-5},
                            {'params': actor.net.classifier.feature_extractor.parameters(), 'lr': 5e-6},
                            {'params': actor.net.classifier.transformer.parameters(), 'lr': 1e-4},
                            {'params': actor.net.bb_regressor.parameters(), 'lr': 1e-4},
                            {'params': actor.net.feature_extractor.layer3.parameters(), 'lr': 2e-6}],
                           lr=2e-5)
    '''

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)
    max_epochs = run_training_params['epoch_num']
    max_save_snapshots = run_training_params['max_save_snapshots']
    # print('------------------------------------------transformer_dimp.py---------------------------------------------------')
    pretrained_model = BASE_DIR + '/pretrained/trdimp_net_ep.pth.tar'
    trainer.train(max_epochs,max_save_snapshots, pretrained_model, load_latest=True, fail_safe=True)
    
