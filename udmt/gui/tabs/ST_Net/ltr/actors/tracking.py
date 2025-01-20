
import os

import cv2
import numpy as np

from . import BaseActor
import torch
# from pytracking import dcf
from udmt.gui.tabs.ST_Net.pytracking.libs import dcf
from udmt.gui.tabs.ST_Net.ltr.data import processing_utils as prutils
# import ltr.data.processing_utils as prutils
debug_flag = False
class DiMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):

        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        # target_scores, iou_pred = self.net(train_imgs=data['train_images'],
        #                                    test_imgs=data['test_images'],
        #                                    train_bb=data['train_anno'],
        #                                    train_label=data['train_label'],  ##
        #                                    test_proposals=data['test_proposals'])
        target_scores = self.net(train_imgs=data['train_images'], # (3,10,3,352,352)
                                           test_imgs=data['test_images'], # (3,10,3,352,352)
                                           train_bb=data['train_anno']) # (3,10,4))
        debug_forward = True
        debug_backward = False


        # target_scores length:6 (3,10,23,23)
        unsupervised_flag = True

        if unsupervised_flag:
            # translation_vec_list = torch.zeros_like(data['train_anno'])
            predict_test_anno_list = torch.zeros_like(data['train_anno'])
            predict_test_label_list = torch.zeros_like(data['train_label'])
            train_label_back_list = torch.zeros_like(data['test_label'])
            target_scores_iter = target_scores[-1]
            # print(target_scores_iter[0][0])
            # print(target_scores_iter[0,0])
            for test_img_num in range(target_scores_iter.shape[0]):
                for batch_num in range(target_scores_iter.shape[1]):
                    if debug_forward:
                        if debug_flag:
                            debug_save_path = './save_tmp_trdimp'
                            if not os.path.exists(debug_save_path):
                                os.makedirs(debug_save_path)

                        if batch_num == 0:

                            save_patch = data['train_images'][test_img_num,batch_num].cpu().numpy()
                            save_patch = np.transpose(save_patch, (1, 2, 0))
                            save_patch = save_patch * 255
                            if debug_flag:
                                cv2.imwrite(debug_save_path+'/patch_train_'+str(test_img_num)+'.jpg', save_patch)

                            save_patch = data['test_images'][test_img_num,batch_num].cpu().numpy()
                            save_patch = np.transpose(save_patch, (1, 2, 0))
                            save_patch = save_patch * 255
                            if debug_flag:
                                cv2.imwrite(debug_save_path+'/patch_test_'+str(test_img_num)+'.jpg', save_patch)


                    translation_vec, _, _ = localize_target(data, target_scores_iter[test_img_num,batch_num])
                    new_pos = translation_vec+data['settings'].output_sz/2
                    new_pos = new_pos[[1,0]]
                    test_size = torch.tensor(data['test_anno'][test_img_num,batch_num][2:],device='cpu')
                    predict_test_anno = torch.cat([new_pos-test_size/2,test_size], dim=0)
                    # print('gt pos:',torch.tensor(data['test_anno'][test_img_num,batch_num][:2]+1/2*data['test_anno'][test_img_num,batch_num][2:],device='cpu').numpy())
                    # print('predict pos:',new_pos)
                    # print('predict box:',predict_test_anno)
                    predict_test_anno = torch.clamp(predict_test_anno, 0, data['settings'].output_sz)
                    predict_test_label = generate_train_label_from_bbbox(data,predict_test_anno.unsqueeze(0))
                    predict_test_label = predict_test_label.squeeze(0)
                    predict_test_anno_list[test_img_num,batch_num,...] = predict_test_anno
                    predict_test_label_list[test_img_num,batch_num,...] = predict_test_label
                train_label_back = generate_label_function_from_bbbox_test(data, torch.tensor(data['train_anno'][test_img_num],device='cpu'))
                train_label_back_list[test_img_num,...] = train_label_back

            backward_track_train_scores = self.net(train_imgs=data['test_images'],
                                               test_imgs=data['train_images'],
                                               train_bb=predict_test_anno_list)
            if debug_backward:
                backward_target_scores_iter = backward_track_train_scores[-1]

                for test_img_num in range(backward_target_scores_iter.shape[0]):
                    for batch_num in range(backward_target_scores_iter.shape[1]):
                        if debug_backward:
                            if debug_flag:
                                debug_save_path = './save_tmp_trdimp'
                                if not os.path.exists(debug_save_path):
                                    os.makedirs(debug_save_path)
                            save_patch = data['train_images'][test_img_num,batch_num].cpu().numpy()
                            save_patch = np.transpose(save_patch, (1, 2, 0))
                            save_patch = save_patch * 255
                            if debug_flag:
                                cv2.imwrite(debug_save_path+'/patch_train_back.jpg', save_patch)

                            save_patch = data['test_images'][test_img_num,batch_num].cpu().numpy()
                            save_patch = np.transpose(save_patch, (1, 2, 0))
                            save_patch = save_patch * 255
                            if debug_flag:
                                cv2.imwrite(debug_save_path+'/patch_test_back.jpg', save_patch)


                        translation_vec, _, _ = localize_target(data, backward_target_scores_iter[test_img_num,batch_num])
                        new_pos = translation_vec+data['settings'].output_sz/2
                        new_pos = new_pos[[1,0]]
                        test_size = torch.tensor(data['train_anno'][test_img_num,batch_num][2:],device='cpu')
                        predict_test_anno = torch.cat([new_pos-test_size/2,test_size], dim=0)
                        print('gt pos:',torch.tensor(data['train_anno'][test_img_num,batch_num][:2]+1/2*data['train_anno'][test_img_num,batch_num][2:],device='cpu').numpy())
                        print('predict pos:',new_pos)
                        print('predict box:',predict_test_anno)

            loss_target_classifier_unsuper = 0
            loss_test_init_clf_unsuper = 0
            loss_test_iter_clf_unsuper = 0
            if 'test_clf' in self.loss_weight.keys():
                # Classification losses for the different optimization iterations
                clf_losses_test_unsuper = [self.objective['test_clf'](s, train_label_back_list, data['train_anno']) for s in backward_track_train_scores]

                # Loss of the final filter
                clf_loss_test_unsuper = clf_losses_test_unsuper[-1]
                loss_target_classifier_unsuper = self.loss_weight['test_clf'] * clf_loss_test_unsuper

                # Loss for the initial filter iteration
                if 'test_init_clf' in self.loss_weight.keys():
                    loss_test_init_clf_unsuper = self.loss_weight['test_init_clf'] * clf_losses_test_unsuper[0]

                # Loss for the intermediate filter iterations
                if 'test_iter_clf' in self.loss_weight.keys():
                    test_iter_weights = self.loss_weight['test_iter_clf']
                    if isinstance(test_iter_weights, list):
                        loss_test_iter_clf_unsuper = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test_unsuper[1:-1])])
                    else:
                        loss_test_iter_clf_unsuper = (test_iter_weights / (len(clf_losses_test_unsuper) - 2)) * sum(clf_losses_test_unsuper[1:-1])
            print('unsuper:',loss_target_classifier_unsuper,loss_test_init_clf_unsuper,loss_test_iter_clf_unsuper)
            loss_unsuper = loss_target_classifier_unsuper + loss_test_init_clf_unsuper + loss_test_iter_clf_unsuper
            bb_ce = 0
            loss_bb_ce = 0
            if torch.isinf(loss_unsuper) or torch.isnan(loss_unsuper):
                raise Exception('ERROR: Loss was nan or inf!!!')

            # Log stats
            stats_unsuper = {'Loss/total': loss_unsuper.item(),
                     'Loss/bb_ce': bb_ce,
                     'Loss/loss_bb_ce': loss_bb_ce}
            if 'test_clf' in self.loss_weight.keys():
                stats_unsuper['Loss/target_clf'] = loss_target_classifier_unsuper.item()
            if 'test_init_clf' in self.loss_weight.keys():
                stats_unsuper['Loss/test_init_clf'] = loss_test_init_clf_unsuper.item()
            if 'test_iter_clf' in self.loss_weight.keys():
                stats_unsuper['Loss/test_iter_clf'] = loss_test_iter_clf_unsuper.item()

            if 'test_clf' in self.loss_weight.keys():
                stats_unsuper['ClfTrain/test_loss'] = clf_loss_test_unsuper.item()
                if len(clf_losses_test_unsuper) > 0:
                    stats_unsuper['ClfTrain/test_init_loss'] = clf_losses_test_unsuper[0].item()
                    if len(clf_losses_test_unsuper) > 2:
                        stats_unsuper['ClfTrain/test_iter_loss'] = sum(clf_losses_test_unsuper[1:-1]).item() / (len(clf_losses_test_unsuper) - 2)


        else:
            # translation_vec, _, _ = localize_target(data, target_scores_iter[0][0])
            # print(translation_vec)
            # translation_vec = torch.cat([translation_vec,torch.Tensor([0,0])], dim=0)
            # predict_test_anno = data['test_anno'][0][0].cpu() + translation_vec
            # predict_test_anno = predict_test_anno.unsqueeze(0)
            # predict_test_label = generate_train_label_from_bbbox(data,predict_test_anno)



            # Reshape bb reg variables
            # is_valid = data['test_anno'][:, :, 0] < 99999.0
            # # bb_scores = bb_scores[is_valid, :]
            # proposal_density = data['proposal_density'][is_valid, :]
            # gt_density = data['gt_density'][is_valid, :]

            # Compute loss
            # bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
            # loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

            # If standard DiMP classifier is used


            loss_target_classifier = 0
            loss_test_init_clf = 0
            loss_test_iter_clf = 0
            if 'test_clf' in self.loss_weight.keys():
                # Classification losses for the different optimization iterations
                clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

                # Loss of the final filter
                clf_loss_test = clf_losses_test[-1]
                loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

                # Loss for the initial filter iteration
                if 'test_init_clf' in self.loss_weight.keys():
                    loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

                # Loss for the intermediate filter iterations
                if 'test_iter_clf' in self.loss_weight.keys():
                    test_iter_weights = self.loss_weight['test_iter_clf']
                    if isinstance(test_iter_weights, list):
                        loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                    else:
                        loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])


            # If PrDiMP classifier is used
            # loss_clf_ce = 0
            # loss_clf_ce_init = 0
            # loss_clf_ce_iter = 0
            # if 'clf_ce' in self.loss_weight.keys():
            #     # Classification losses for the different optimization iterations
            #     clf_ce_losses = [self.objective['clf_ce'](s, data['test_label_density'], grid_dim=(-2,-1)) for s in target_scores]
            #
            #     # Loss of the final filter
            #     clf_ce = clf_ce_losses[-1]
            #     loss_clf_ce = self.loss_weight['clf_ce'] * clf_ce
            #
            #     # Loss for the initial filter iteration
            #     if 'clf_ce_init' in self.loss_weight.keys():
            #         loss_clf_ce_init = self.loss_weight['clf_ce_init'] * clf_ce_losses[0]
            #
            #     # Loss for the intermediate filter iterations
            #     if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            #         test_iter_weights = self.loss_weight['clf_ce_iter']
            #         if isinstance(test_iter_weights, list):
            #             loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
            #         else:
            #             loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])

            # Total loss
            # delete bb module
            bb_ce = 0
            loss_bb_ce = 0

            loss = loss_bb_ce + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

            if torch.isinf(loss) or torch.isnan(loss):
                raise Exception('ERROR: Loss was nan or inf!!!')

            # Log stats
            stats = {'Loss/total': loss.item(),
                     'Loss/bb_ce': bb_ce,
                     'Loss/loss_bb_ce': loss_bb_ce}
            if 'test_clf' in self.loss_weight.keys():
                stats['Loss/target_clf'] = loss_target_classifier.item()
            if 'test_init_clf' in self.loss_weight.keys():
                stats['Loss/test_init_clf'] = loss_test_init_clf.item()
            if 'test_iter_clf' in self.loss_weight.keys():
                stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
            # if 'clf_ce' in self.loss_weight.keys():
            #     stats['Loss/clf_ce'] = loss_clf_ce.item()
            # if 'clf_ce_init' in self.loss_weight.keys():
            #     stats['Loss/clf_ce_init'] = loss_clf_ce_init.item()
            # if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            #     stats['Loss/clf_ce_iter'] = loss_clf_ce_iter.item()

            if 'test_clf' in self.loss_weight.keys():
                stats['ClfTrain/test_loss'] = clf_loss_test.item()
                if len(clf_losses_test) > 0:
                    stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                    if len(clf_losses_test) > 2:
                        stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

            # if 'clf_ce' in self.loss_weight.keys():
            #     stats['ClfTrain/clf_ce'] = clf_ce.item()
            #     if len(clf_ce_losses) > 0:
            #         stats['ClfTrain/clf_ce_init'] = clf_ce_losses[0].item()
            #         if len(clf_ce_losses) > 2:
            #             stats['ClfTrain/clf_ce_iter'] = sum(clf_ce_losses[1:-1]).item() / (len(clf_ce_losses) - 2)
        if unsupervised_flag:
            return loss_unsuper, stats_unsuper
        else:
            return loss, stats


def localize_target(data,scores):
    """Run the target localization."""
    scores = scores.unsqueeze(0)
    # scores = scores.squeeze(1)
    ''' 没有运行到的代码
    preprocess_method = self.params.get('score_preprocess', 'none')
    if preprocess_method == 'none':
        pass
    elif preprocess_method == 'exp':
        print('!')
        scores = scores.exp()
    elif preprocess_method == 'softmax':
        print('!!')
        reg_val = getattr(self.net.classifier.filter_optimizer, 'softmax_reg', None)
        scores_view = scores.view(scores.shape[0], -1)
        scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)
        scores = scores_softmax.view(scores.shape)
    else:
        raise Exception('Unknown score_preprocess in params.')

    score_filter_ksz = self.params.get('score_filter_ksz', 1)
    if score_filter_ksz > 1:
        print('!!!')
        assert score_filter_ksz % 2 == 1
        kernel = scores.new_ones(1,1,score_filter_ksz,score_filter_ksz)
        scores = F.conv2d(scores.view(-1,1,*scores.shape[-2:]), kernel, padding=score_filter_ksz//2).view(scores.shape)
    '''
    # change

    # if self.params.get('advanced_localization', False):
    #     return self.localize_advanced(scores, sample_pos, sample_scales) #bad after comment

    # Get maximum
    score_sz = torch.Tensor(list(scores.shape[-2:]))
    score_center = (score_sz - 1)/2
    max_score, max_disp = dcf.max2d(scores)
    _, scale_ind = torch.max(max_score, dim=0)
    max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
    target_disp = max_disp - score_center

    # Compute translation vector and scale change factor

    img_support_sz = torch.Tensor([data['settings'].output_sz, data['settings'].output_sz])
    kernel_size = torch.Tensor([data['settings'].target_filter_sz, data['settings'].target_filter_sz])

    output_sz = score_sz - (kernel_size + 1) % 2
    translation_vec = target_disp * (img_support_sz / output_sz)

    return translation_vec, scale_ind, scores
def generate_label_function_from_bbbox_test(data,target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), data['settings'].sigma_factor_label,
                                                      data['settings'].target_filter_sz,
                                                      data['settings'].feature_sz, data['settings'].output_sz,
                                                      end_pad_if_even=True)

        return gauss_label

def generate_train_label_from_bbbox(data,target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), 0.1,
                                                      data['settings'].target_filter_sz,
                                                      data['settings'].feature_sz, data['settings'].output_sz,
                                                      end_pad_if_even=False)
        return gauss_label
class KLDiMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        # target_scores, iou_pred = self.net(train_imgs=data['train_images'],
        #                                    test_imgs=data['test_images'],
        #                                    train_bb=data['train_anno'],
        #                                    train_label=data['train_label'],  ##
        #                                    test_proposals=data['test_proposals'])
        target_scores = self.net(train_imgs=data['train_images'], # (3,10,3,352,352)
                                           test_imgs=data['test_images'], # (3,10,3,352,352)
                                           train_bb=data['train_anno'], # (3,10,4)
                                           train_label=data['train_label'])
        debug_forward = False
        debug_backward = False


        # target_scores length:6 (3,10,23,23)
        unsupervised_flag = True

        if unsupervised_flag:
            # translation_vec_list = torch.zeros_like(data['train_anno'])
            predict_test_anno_list = torch.zeros_like(data['train_anno'])
            predict_test_label_list = torch.zeros_like(data['train_label'])
            train_label_back_list = torch.zeros_like(data['test_label'])
            target_scores_iter = target_scores[-1]
            # print(target_scores_iter[0][0])
            # print(target_scores_iter[0,0])
            for test_img_num in range(target_scores_iter.shape[0]):
                for batch_num in range(target_scores_iter.shape[1]):

                    if debug_forward:
                        debug_save_path = './save_tmp_trdimp'
                        if not os.path.exists(debug_save_path):
                            os.makedirs(debug_save_path)
                        save_patch = data['train_images'][test_img_num,batch_num].cpu().numpy()
                        save_patch = np.transpose(save_patch, (1, 2, 0))
                        save_patch = save_patch * 255
                        cv2.imwrite(debug_save_path+'/patch_train.jpg', save_patch)

                        save_patch = data['test_images'][test_img_num,batch_num].cpu().numpy()
                        save_patch = np.transpose(save_patch, (1, 2, 0))
                        save_patch = save_patch * 255
                        cv2.imwrite(debug_save_path+'/patch_test.jpg', save_patch)


                    translation_vec, _, _ = localize_target(data, target_scores_iter[test_img_num,batch_num])
                    new_pos = translation_vec+data['settings'].output_sz/2
                    new_pos = new_pos[[1,0]]
                    # test_size = torch.tensor(data['test_anno'][test_img_num,batch_num][2:],device='cpu')
                    test_size = data['test_anno'][test_img_num, batch_num][2:].clone().detach().to('cpu')#######debug in 241223
                    predict_test_anno = torch.cat([new_pos-test_size/2,test_size], dim=0)
                    # print('gt pos:',torch.tensor(data['test_anno'][test_img_num,batch_num][:2]+1/2*data['test_anno'][test_img_num,batch_num][2:],device='cpu').numpy())
                    # print('predict pos:',new_pos)
                    # print('predict box:',predict_test_anno)
                    predict_test_anno = torch.clamp(predict_test_anno, 0, data['settings'].output_sz)
                    predict_test_label = generate_train_label_from_bbbox(data,predict_test_anno.unsqueeze(0))
                    predict_test_label = predict_test_label.squeeze(0)
                    predict_test_anno_list[test_img_num,batch_num,...] = predict_test_anno
                    predict_test_label_list[test_img_num,batch_num,...] = predict_test_label
                # train_label_back = generate_label_function_from_bbbox_test(data, torch.tensor(data['train_anno'][test_img_num],device='cpu'))
                train_label_back = generate_label_function_from_bbbox_test(data, data['train_anno'][test_img_num].clone().detach().to('cpu'))#######debug in 241223
                train_label_back_list[test_img_num,...] = train_label_back

            backward_track_train_scores = self.net(train_imgs=data['test_images'],
                                               test_imgs=data['train_images'],
                                               train_bb=predict_test_anno_list,
                                               train_label=predict_test_label_list)
            if debug_backward:
                backward_target_scores_iter = backward_track_train_scores[-1]

                for test_img_num in range(backward_target_scores_iter.shape[0]):
                    for batch_num in range(backward_target_scores_iter.shape[1]):
                        if debug_backward:
                            debug_save_path = './save_tmp_trdimp'
                            if not os.path.exists(debug_save_path):
                                os.makedirs(debug_save_path)
                            save_patch = data['train_images'][test_img_num,batch_num].cpu().numpy()
                            save_patch = np.transpose(save_patch, (1, 2, 0))
                            save_patch = save_patch * 255
                            cv2.imwrite(debug_save_path+'/patch_train_back.jpg', save_patch)

                            save_patch = data['test_images'][test_img_num,batch_num].cpu().numpy()
                            save_patch = np.transpose(save_patch, (1, 2, 0))
                            save_patch = save_patch * 255
                            cv2.imwrite(debug_save_path+'/patch_test_back.jpg', save_patch)


                        translation_vec, _, _ = localize_target(data, backward_target_scores_iter[test_img_num,batch_num])
                        new_pos = translation_vec+data['settings'].output_sz/2
                        new_pos = new_pos[[1,0]]
                        test_size = torch.tensor(data['train_anno'][test_img_num,batch_num][2:],device='cpu')
                        predict_test_anno = torch.cat([new_pos-test_size/2,test_size], dim=0)
                        print('gt pos:',torch.tensor(data['train_anno'][test_img_num,batch_num][:2]+1/2*data['train_anno'][test_img_num,batch_num][2:],device='cpu').numpy())
                        print('predict pos:',new_pos)
                        print('predict box:',predict_test_anno)

            loss_target_classifier_unsuper = 0
            loss_test_init_clf_unsuper = 0
            loss_test_iter_clf_unsuper = 0
            if 'test_clf' in self.loss_weight.keys():
                # Classification losses for the different optimization iterations
                clf_losses_test_unsuper = [self.objective['test_clf'](s, train_label_back_list, data['train_anno']) for s in backward_track_train_scores]

                # Loss of the final filter
                clf_loss_test_unsuper = clf_losses_test_unsuper[-1]
                loss_target_classifier_unsuper = self.loss_weight['test_clf'] * clf_loss_test_unsuper

                # Loss for the initial filter iteration
                if 'test_init_clf' in self.loss_weight.keys():
                    loss_test_init_clf_unsuper = self.loss_weight['test_init_clf'] * clf_losses_test_unsuper[0]

                # Loss for the intermediate filter iterations
                if 'test_iter_clf' in self.loss_weight.keys():
                    test_iter_weights = self.loss_weight['test_iter_clf']
                    if isinstance(test_iter_weights, list):
                        loss_test_iter_clf_unsuper = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test_unsuper[1:-1])])
                    else:
                        loss_test_iter_clf_unsuper = (test_iter_weights / (len(clf_losses_test_unsuper) - 2)) * sum(clf_losses_test_unsuper[1:-1])
            # print('unsuper:',loss_target_classifier_unsuper,loss_test_init_clf_unsuper,loss_test_iter_clf_unsuper)
            loss_unsuper = loss_target_classifier_unsuper + loss_test_init_clf_unsuper + loss_test_iter_clf_unsuper
            bb_ce = 0
            loss_bb_ce = 0
            if torch.isinf(loss_unsuper) or torch.isnan(loss_unsuper):
                raise Exception('ERROR: Loss was nan or inf!!!')

            # Log stats
            stats_unsuper = {'Loss/total': loss_unsuper.item(),
                     'Loss/bb_ce': bb_ce,
                     'Loss/loss_bb_ce': loss_bb_ce}
            if 'test_clf' in self.loss_weight.keys():
                stats_unsuper['Loss/target_clf'] = loss_target_classifier_unsuper.item()
            if 'test_init_clf' in self.loss_weight.keys():
                stats_unsuper['Loss/test_init_clf'] = loss_test_init_clf_unsuper.item()
            if 'test_iter_clf' in self.loss_weight.keys():
                stats_unsuper['Loss/test_iter_clf'] = loss_test_iter_clf_unsuper.item()

            if 'test_clf' in self.loss_weight.keys():
                stats_unsuper['ClfTrain/test_loss'] = clf_loss_test_unsuper.item()
                if len(clf_losses_test_unsuper) > 0:
                    stats_unsuper['ClfTrain/test_init_loss'] = clf_losses_test_unsuper[0].item()
                    if len(clf_losses_test_unsuper) > 2:
                        stats_unsuper['ClfTrain/test_iter_loss'] = sum(clf_losses_test_unsuper[1:-1]).item() / (len(clf_losses_test_unsuper) - 2)


        else:
            # translation_vec, _, _ = localize_target(data, target_scores_iter[0][0])
            # print(translation_vec)
            # translation_vec = torch.cat([translation_vec,torch.Tensor([0,0])], dim=0)
            # predict_test_anno = data['test_anno'][0][0].cpu() + translation_vec
            # predict_test_anno = predict_test_anno.unsqueeze(0)
            # predict_test_label = generate_train_label_from_bbbox(data,predict_test_anno)



            # Reshape bb reg variables
            # is_valid = data['test_anno'][:, :, 0] < 99999.0
            # # bb_scores = bb_scores[is_valid, :]
            # proposal_density = data['proposal_density'][is_valid, :]
            # gt_density = data['gt_density'][is_valid, :]

            # Compute loss
            # bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
            # loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

            # If standard DiMP classifier is used


            loss_target_classifier = 0
            loss_test_init_clf = 0
            loss_test_iter_clf = 0
            if 'test_clf' in self.loss_weight.keys():
                # Classification losses for the different optimization iterations
                clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

                # Loss of the final filter
                clf_loss_test = clf_losses_test[-1]
                loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

                # Loss for the initial filter iteration
                if 'test_init_clf' in self.loss_weight.keys():
                    loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

                # Loss for the intermediate filter iterations
                if 'test_iter_clf' in self.loss_weight.keys():
                    test_iter_weights = self.loss_weight['test_iter_clf']
                    if isinstance(test_iter_weights, list):
                        loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                    else:
                        loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])


            # If PrDiMP classifier is used
            # loss_clf_ce = 0
            # loss_clf_ce_init = 0
            # loss_clf_ce_iter = 0
            # if 'clf_ce' in self.loss_weight.keys():
            #     # Classification losses for the different optimization iterations
            #     clf_ce_losses = [self.objective['clf_ce'](s, data['test_label_density'], grid_dim=(-2,-1)) for s in target_scores]
            #
            #     # Loss of the final filter
            #     clf_ce = clf_ce_losses[-1]
            #     loss_clf_ce = self.loss_weight['clf_ce'] * clf_ce
            #
            #     # Loss for the initial filter iteration
            #     if 'clf_ce_init' in self.loss_weight.keys():
            #         loss_clf_ce_init = self.loss_weight['clf_ce_init'] * clf_ce_losses[0]
            #
            #     # Loss for the intermediate filter iterations
            #     if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            #         test_iter_weights = self.loss_weight['clf_ce_iter']
            #         if isinstance(test_iter_weights, list):
            #             loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
            #         else:
            #             loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])

            # Total loss
            # delete bb module
            bb_ce = 0
            loss_bb_ce = 0

            loss = loss_bb_ce + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

            if torch.isinf(loss) or torch.isnan(loss):
                raise Exception('ERROR: Loss was nan or inf!!!')

            # Log stats
            stats = {'Loss/total': loss.item(),
                     'Loss/bb_ce': bb_ce,
                     'Loss/loss_bb_ce': loss_bb_ce}
            if 'test_clf' in self.loss_weight.keys():
                stats['Loss/target_clf'] = loss_target_classifier.item()
            if 'test_init_clf' in self.loss_weight.keys():
                stats['Loss/test_init_clf'] = loss_test_init_clf.item()
            if 'test_iter_clf' in self.loss_weight.keys():
                stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
            # if 'clf_ce' in self.loss_weight.keys():
            #     stats['Loss/clf_ce'] = loss_clf_ce.item()
            # if 'clf_ce_init' in self.loss_weight.keys():
            #     stats['Loss/clf_ce_init'] = loss_clf_ce_init.item()
            # if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            #     stats['Loss/clf_ce_iter'] = loss_clf_ce_iter.item()

            if 'test_clf' in self.loss_weight.keys():
                stats['ClfTrain/test_loss'] = clf_loss_test.item()
                if len(clf_losses_test) > 0:
                    stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                    if len(clf_losses_test) > 2:
                        stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

            # if 'clf_ce' in self.loss_weight.keys():
            #     stats['ClfTrain/clf_ce'] = clf_ce.item()
            #     if len(clf_ce_losses) > 0:
            #         stats['ClfTrain/clf_ce_init'] = clf_ce_losses[0].item()
            #         if len(clf_ce_losses) > 2:
            #             stats['ClfTrain/clf_ce_iter'] = sum(clf_ce_losses[1:-1]).item() / (len(clf_ce_losses) - 2)
        if unsupervised_flag:
            return loss_unsuper, stats_unsuper
        else:
            return loss, stats
