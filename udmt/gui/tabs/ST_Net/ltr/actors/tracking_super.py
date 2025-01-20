
from . import BaseActor
import torch
from pytracking import dcf
import ltr.data.processing_utils as prutils
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
        target_scores, iou_pred = self.net(train_imgs=data['train_images'],
                                           test_imgs=data['test_images'],
                                           train_bb=data['train_anno'],
                                           train_label=data['train_label'],  ##
                                           test_proposals=data['test_proposals'])

        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Compute loss for ATOM IoUNet
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'])

        # Loss for the initial filter iteration
        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a*b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        stats['ClfTrain/test_loss'] = clf_loss_test.item()
        if len(clf_losses_test) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
            if len(clf_losses_test) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

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
        target_scores, bb_scores = self.net(train_imgs=data['train_images'], # 3,10,3,352,352
                                            test_imgs=data['test_images'], # 3,10,3,352,352
                                            train_bb=data['train_anno'], # 3,10,4
                                            train_label=data['train_label'],  # 3,10,22,22
                                            test_proposals=data['test_proposals'])
        # target_scores length:6 (1,10,23,23)
        target_scores_iter = target_scores[-1]
        translation_vec, _ ,_ = localize_target(data,target_scores_iter[0][0])
        print(translation_vec)
        predict_test_anno = data['test_anno'][0][0][:2] + translation_vec
        

        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

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
        loss = loss_bb_ce + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
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

        return loss, stats
