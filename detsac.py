from utils import \
    options, initialisation, sampling, backward, visualisation, evaluation, residual_functions, inlier_counting, metrics, postprocessing, loss_functions
import torch
import torch.nn.functional as F
import time
from utils.prepare_labels import prepare_labels
from tqdm import tqdm

opt = options.get_options()

initialisation.seeds(opt)

ckpt_dir, log = initialisation.setup_logging_and_checkpointing(opt)

model, optimizer, scheduler, device = initialisation.get_model(opt)

datasets = initialisation.get_dataset(opt)



dataloaders = initialisation.get_dataloader(opt, datasets, shuffle_all=False)

for epoch in range(opt.epochs):

    print("epoch %d / %d" % (epoch + 1, opt.epochs))
    epoch_start = time.time()


    # lll = [ fff for i, fff in enumerate(dataloaders['train']) ]
    # ddds = []
    # for i in range(len(dataloaders['train'].dataset)):
    #     item = dataloaders['train'].dataset[i]
    #     ddds.append((
    #         item[1].sum(), # X
    #         item[2].sum(), # gt_labels
    #         item[-1].sum()  # residual_labels
    #     ))
    # ds_checksum = [(x[0].sum(), x[1].sum(), x[-1].sum()) for x in dataloaders['train']]

    # ddds = torch.tensor(ddds)
    # ds_checksum = torch.tensor(ds_checksum)

    # print(opt.num_workers)
    # print(ddds.sum(axis=0))
    # print(ds_checksum.sum(axis=0))
    # print((np.array(dataloaders['train'].dataset.item_seeds) % 10000).sum() % 10000)

    # breakpoint()

    for mode in opt.modes:

        assert not (dataloaders[mode] is None), "no dataloader for %s available" % mode

        print("mode: %s" % mode)

        ds_checksum = []

        if mode == "train":
            model.train()
            all_loss = 0
            all_loss_matched_classes = 0
            all_loss_latent_code = 0
            all_loss_unmatched_classes = 0
            
            for batch_idx, (features, X, gt_labels, gt_models, image, image_size, mask, residual_labels) in enumerate(tqdm(dataloaders[mode])):

                gt_labels, residual_labels = prepare_labels(gt_labels, residual_labels)
                
                ds_checksum.append([
                    X.sum(),
                    gt_labels.sum(),
                    residual_labels.sum()
                ])

                X = X.to(device)
                features = features.to(device)
                gt_labels = gt_labels.to(device)
                gt_models = gt_models.to(device)
                image_size = image_size.to(device)
                mask = mask.to(device)
                residual_labels = residual_labels.to(device)
                
                optimizer.zero_grad()
                
                latent_code_output, point_output = model(features, mask) # (B, O, 1), (B, O, N)
                
                with torch.no_grad():
                    assignments = loss_functions.hungarian_logit_matches(point_output, gt_labels)
                
                loss_matched_classes = loss_functions.compute_matched_loss(point_output, gt_labels, assignments, residual_labels)
                loss_unmatched_classes = loss_functions.compute_unmatched_loss(point_output, gt_labels, assignments)
                loss_latent_code = loss_functions.compute_latent_code_loss(latent_code_output, assignments)
                
                loss = loss_matched_classes + loss_latent_code + loss_unmatched_classes
                
                loss.backward()
                optimizer.step()
                
                all_loss_matched_classes += loss_matched_classes
                all_loss_latent_code += loss_latent_code
                all_loss_unmatched_classes += loss_unmatched_classes
                
                all_loss += loss
                
            print(f'Train Loss: {all_loss}')
            print(f'    Train Matched Classes Loss: {all_loss_matched_classes}')
            print(f'    Train Latent Codes Loss: {all_loss_latent_code}')
            print(f'    Train Unmatched Classes Loss: {all_loss_unmatched_classes}')
            
            scheduler.step()
            dataloaders[mode].dataset.step()

            
        else:
            model.eval()
            
            eval_metrics = {"loss": [], "time": [], "total_time": [], "loss_matched_classes":[], "loss_latent_code":[], "loss_unmatched_classes":[]}
            wandb_log_data = {}
            all_loss = 0
            all_loss_matched_classes = 0
            all_loss_latent_code = 0
            all_loss_unmatched_classes = 0 
            
            total_start = time.time()

            for batch_idx, (features, X, gt_labels, gt_models, image, image_size, mask, residual_labels) in enumerate(tqdm(dataloaders[mode])):

                gt_labels, residual_labels = prepare_labels(gt_labels, residual_labels)

                ds_checksum.append([
                    X.sum(),
                    gt_labels.sum(),
                    residual_labels.sum()
                ])

                for run_idx in range(opt.runcount):

                    X = X.to(device)
                    features = features.to(device)
                    gt_labels = gt_labels.to(device)
                    gt_models = gt_models.to(device)
                    image_size = image_size.to(device)
                    mask = mask.to(device)
                    residual_labels = residual_labels.to(device)
                    
                    optimizer.zero_grad()

                    start_time = time.time()

                    with torch.no_grad():
                        latent_code_classifier_output, point_output = model(features, mask) # (B, O, 1), (B, O, N)
                        
                        latent_code_classifier_output, idx_sort = latent_code_classifier_output.sort(dim=1, descending=True)
                        point_output = point_output.gather(1, idx_sort.expand(-1, -1, point_output.size(-1)))
                        
                        assignments = loss_functions.hungarian_logit_matches(point_output, gt_labels)
                        loss_matched_classes = loss_functions.compute_matched_loss(point_output, gt_labels, assignments, residual_labels)
                        loss_latent_code = loss_functions.compute_latent_code_loss(latent_code_classifier_output, assignments)
                        loss_unmatched_classes = loss_functions.compute_unmatched_loss(point_output, gt_labels, assignments)
                        
                        loss = loss_matched_classes + loss_latent_code + loss_unmatched_classes
                        
                        all_loss_matched_classes += loss_matched_classes
                        all_loss_latent_code += loss_latent_code
                        all_loss_unmatched_classes += loss_unmatched_classes
                        all_loss += loss
                        
                        eval_metrics['loss'].append(loss.item())
                        eval_metrics['loss_matched_classes'].append(loss_matched_classes.item())
                        eval_metrics['loss_latent_code'].append(loss_latent_code.item())
                        eval_metrics['loss_unmatched_classes'].append(loss_unmatched_classes.item())
                        # the sampling part
                        ng = torch.transpose(F.sigmoid(point_output), 1, 2) # B, N, O == M
                        # normalizer1 = torch.sum(ng, dim=-1, keepdim=True)
                        normalizer2 = torch.sum(ng, dim=-2, keepdim=True)
                        inlier_weights = ng
                        # We could interpret the sample weights in two ways: probability of each point is independent to each class or normalize it with the sum of probabilities of each class
                        # inlier_weights = ng/normalizer1
                        sample_weights = ng/normalizer2 # B, N, O == M

                        # breakpoint()
                        
                        for i in range(len(sample_weights)):
                            sample_weights_i = sample_weights[i][None] # 1, N, O == M
                            inlier_weights_i = inlier_weights[i][None] # 1, N, O == M
                            X_i = X[i][None]
                            gt_models_i = gt_models[i][None]
                            gt_labels_i = gt_labels[i][None]
                            image_size_i = image_size[i][None]
                                                        
                            # minimal_sets = sampling.sample_minimal_sets(opt, sample_weights_i, softmax=False, topk=None)
                            opt.topk = max((latent_code_classifier_output > 0).sum(dim=1).max().item(), 1)
                            minimal_sets = sampling.sample_minimal_sets(opt, sample_weights_i, softmax=False, topk=opt.topk)
                            hypotheses = sampling.generate_hypotheses(opt, X_i, minimal_sets)
                            residuals = residual_functions.compute_residuals(opt, X_i, hypotheses)
                            
                            # modified from count_inliers
                            inlier_fun = inlier_counting.inlier_functions[opt.inlier_function](opt.inlier_softness, opt.inlier_threshold)
                            inlier_scores = inlier_fun(residuals)
                            
                            B, N, Mo = inlier_weights_i.size()
                            _, K, S, M, N = residuals.size()
                            weights_e = inlier_weights_i[..., :M].transpose(1, 2).view(B, 1, 1, M, N).expand(B, K, S, M, N)
                            
                            if opt.inlier_counting == "unweighted":
                                inlier_scores_weighted = inlier_scores
                            else:
                                inlier_scores_weighted = inlier_scores * weights_e
                            inlier_scores = inlier_scores_weighted
                            inlier_ratios = inlier_scores_weighted.sum(-1) * 1.0 / N

                            log_p_M_S, sampled_inlier_scores, sampled_hypotheses, sampled_residuals = \
                                sampling.sample_hypotheses(opt, mode, hypotheses, inlier_ratios, inlier_scores, residuals)
                            # log_p_M_S: B, K, H
                            # sampled_inlier_scores: B, K, M, H, N
                            # sampled_hypotheses: B, K, M, H, D
                            # sampled_residuals: B, K, M, H, N

                            if opt.refine:
                                if opt.problem == "vp":
                                    sampled_hypotheses, sampled_residuals, sampled_inlier_scores = \
                                        postprocessing.refinement_with_inliers(opt, X_i, sampled_inlier_scores)

                            # start = time.time()
                            ranked_choices, ranked_inlier_ratios, ranked_hypotheses, ranked_scores, labels, clusters = \
                                postprocessing.ranking_and_clustering(opt, sampled_inlier_scores, sampled_hypotheses,
                                                                    sampled_residuals)
                            # ranked_choices: B, K, M, H
                            # ranked_inlier_ratios: B, K, M, H
                            # ranked_hypotheses: B, K, M, H, D
                            # ranked_scores: B, K, M, H, N
                            # labels: 
                            # clusters: 
                            
                            
                            # ranked_inlier_ratios = sampled_inlier_scores.max(dim=2, keepdim=True)[0].sum(-1)
                            # labels, clusters = postprocessing.assign_cluster_labels_wo_counts(opt, sampled_residuals)
                            # ranked_counts_per_model = sampled_inlier_scores.sum(-1)
                            # labels, clusters = postprocessing.assign_cluster_labels(opt, sampled_residuals, ranked_counts_per_model)
                            
                            # inlier_weights_topk = sampled_residuals[:, :, :opt.topk].transpose(-2, -3)
                            # indices = torch.min(inlier_weights_topk, dim=3, keepdim=True)[1]
                            # clusters = torch.zeros_like(inlier_weights_topk, dtype=torch.bool)
                            # clusters.scatter_(3, indices, True)

                            # end = time.time()
                                
                            duration = (time.time() - start_time) * 1000
                            eval_metrics["time"] += [duration]
                            
                            
                            eval_metrics = evaluation.compute_validation_metrics(opt, eval_metrics, ranked_hypotheses,
                                                ranked_inlier_ratios, gt_models_i, gt_labels_i, X_i, image_size_i, clusters,
                                                run_idx, datasets["inverse_intrinsics"], train=(mode == "train"))
                            # eval_metrics = evaluation.compute_validation_metrics(opt, eval_metrics, sampled_hypotheses,
                            #                     ranked_inlier_ratios, gt_models_i, gt_labels_i, X_i, image_size_i, clusters,
                            #                     run_idx, datasets["inverse_intrinsics"], train=(mode == "train"))   

                            # breakpoint()                   

                    total_duration = (time.time() - total_start) * 1000
                    eval_metrics["total_time"] += [total_duration]
                    total_start = time.time()
                    
                    # if opt.visualise:
                    #     visualisation.save_visualisation_plots(opt, X, ranked_choices, log_inlier_weights,
                    #                                            log_sample_weights, ranked_scores, clusters,
                    #                                            labels, gt_models, gt_labels, image,
                    #                                            dataloaders[mode].dataset, metrics=eval_metrics)

            visualisation.log_wandb(wandb_log_data, eval_metrics, mode, epoch)
            for key, val in wandb_log_data.items():
                if key in ['val/vp_error_auc_1_avg', 'val/vp_error_auc_3_avg', 
                            'val/vp_error_auc_5_avg', 'val/vp_error_auc_10_avg',
                            'val/misclassification_error_avg',
                            'val/geometric_error_avg',
                            
                            'test/vp_error_auc_1_avg', 'test/vp_error_auc_3_avg', 
                            'test/vp_error_auc_5_avg', 'test/vp_error_auc_10_avg',
                            'test/misclassification_error_avg',
                            'test/geometric_error_avg']:
                    print(key, ':', val)
            
            print(f'Eval Loss: {all_loss}')
            print(f'    Eval Matched Classes Loss: {all_loss_matched_classes}')
            print(f'    Eval Latent Codes Loss: {all_loss_latent_code}')
            print(f'    Eval Unmatched Classes Loss: {all_loss_unmatched_classes}')


        ds_checksum = torch.tensor(ds_checksum)
        print(f"{ds_checksum.sum(axis=0) = }")

        # breakpoint()

    if opt.ckpt_mode == "all":
        torch.save(model.state_dict(), '%s/model_weights_%06d.net' % (ckpt_dir, epoch))
        torch.save(optimizer.state_dict(), '%s/optimizer_%06d.net' % (ckpt_dir, epoch))
    elif opt.ckpt_mode == "last":
        torch.save(model.state_dict(), '%s/model_weights.net' % (ckpt_dir))
        torch.save(optimizer.state_dict(), '%s/optimizer.net' % (ckpt_dir))
    
    epoch_end = time.time()
    print('--------------------------------------------------')
    print(f'Epoch time used: {epoch_end - epoch_start}')
