from sklearn.metrics import precision_recall_curve, auc, log_loss
import numpy as np

def logits_to_prob(x):
  return 1.0/(1.0 + np.exp(-x))

def metric_log_loss(y_t, y_p):
  return log_loss(y_t, y_p)

def relative_cross_entropy(yt, yp):
  cross_entropy = log_loss(yt, yp) #cross entropy between ground truth and prediction
  gt = predict_naive(yt) #this is naive prediction based on average positive 
  cross_entropy_naive = log_loss(yt, gt) #cross entropy between ground truth and naive prediction
  rce = (1.0 - cross_entropy / cross_entropy_naive)*100.0
  return (rce, cross_entropy, cross_entropy_naive)

def predict_naive(yt):
  pred_positive = len([x for x in yt if x == 1])  #number of positive samples
  total_samples = len(yt)
  pred = pred_positive / float(total_samples) 
  yp_naive = [pred for _ in yt]

  return yp_naive

def pr_auc(yt, yp):
  prec, recall, thresh = precision_recall_curve(yt, yp)
  prauc = auc(recall, prec)
  return prauc


""" def recsys_compute_prauc(pred, gt):
  prec, recall, thresh = precision_recall_curve(gt, pred)
  prauc = auc(recall, prec)
  return prauc

def recsys_calculate_ctr(gt):
  positive = len([x for x in gt if x == 1])
  ctr = positive/float(len(gt))
  return ctr

def recsys_compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = recsys_calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0

# ground_truth = read_predictions("gt.csv") # will return data in the form (tweet_id, user_id, labed (1 or 0))
# predictions = read_predictions("predictions.csv") # will return data in the form (tweet_id, user_id, prediction) """
