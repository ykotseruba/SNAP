__author__ = 'tsungyi'

import copy
import datetime
import time
from collections import defaultdict

import numpy as np

#from . import mask as maskUtils
import mask as maskUtils

class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm', lrp_size_details = False, verbose=False):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.verbose = verbose
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        # per-image per-category evaluation results [KxAxI] elements
        self.evalImgs = defaultdict(list)
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iouType=iouType, lrp_size_details=lrp_size_details)  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts
        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def log(self, msg):
        if self.verbose:
            print(msg)

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(
                self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(
            list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results
         (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        self.log('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            self.log('useSegm (deprecated) is not None. Running {} evaluation'.
                  format(p.iouType))
        self.log('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId)
                     for imgId in p.imgIds for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [
            evaluateImg(imgId, catId, areaRng, maxDet) for catId in catIds
            for areaRng in p.areaRng for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        self.log('DONE (t={:0.2f}s).'.format(toc - tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) &
                    # (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0) + \
                        np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + \
                        np.max((z, yd - y1), axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area'] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(
            self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        dtIoU = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store
                        # appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
                    dtIoU[tind, dind] = iou
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1]
                      for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T,
                                                                      0)))
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
            'dtIoUs': dtIoU,
        }

    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and
        store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        self.log('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        # -1 for the precision of absent categories
        precision = -np.ones((T, R, K, A, M))
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))
        olrp_loc = -np.ones((K, A, M))
        olrp_fp = -np.ones((K, A, M))
        olrp_fn = -np.ones((K, A, M))
        olrp = -np.ones((K, A, M))
        lrp_opt_thr = -np.ones((K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [
            n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng))
            if a in setA
        ]
        #i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        i_list = p.imgIds
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate(
                        [e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly
                    # different results.
                    # mergesort is used to be consistent as Matlab
                    # implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                          inds]
                    dtIg = np.concatenate(
                        [e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                         inds]
                    dtIoU = np.concatenate(
                        [e['dtIoUs'][:, 0:maxDet] for e in E], axis=1)[:, inds]

                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm),
                                         np.logical_not(dtIg))

                    dtIoU = np.multiply(dtIoU, tps)
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R, ))
                        ss = np.zeros((R, ))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization
                        # for accessing elements use python array
                        # gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except BaseException:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)

                    # oLRP and Opt.Thr. Computation
                    tp_num = np.cumsum(tps[0, :])
                    fp_num = np.cumsum(fps[0, :])
                    fn_num = npig - tp_num
                    # If there is detection
                    if tp_num.shape[0] > 0:
                        # There is some TPs
                        if tp_num[-1] > 0:
                            total_loc = tp_num - np.cumsum(dtIoU[0, :])
                            lrps = (total_loc / (1 - _pe.iouThrs[0]) + fp_num +
                                    fn_num) / (tp_num + fp_num + fn_num)
                            opt_pos_idx = np.argmin(lrps)
                            olrp[k, a, m] = lrps[opt_pos_idx]
                            olrp_loc[k, a, m] = total_loc[opt_pos_idx] / \
                                tp_num[opt_pos_idx]
                            olrp_fp[k, a, m] = fp_num[opt_pos_idx] / \
                                (tp_num[opt_pos_idx] + fp_num[opt_pos_idx])
                            olrp_fn[k, a, m] = fn_num[opt_pos_idx] / npig
                            lrp_opt_thr[k, a, m] = dtScoresSorted[opt_pos_idx]
                        # There is No TP
                        else:
                            olrp_loc[k, a, m] = np.nan
                            olrp_fp[k, a, m] = np.nan
                            olrp_fn[k, a, m] = 1.
                            olrp[k, a, m] = 1.
                            lrp_opt_thr[k, a, m] = np.nan
                    # No detection
                    else:
                        olrp_loc[k, a, m] = np.nan
                        olrp_fp[k, a, m] = np.nan
                        olrp_fn[k, a, m] = 1.
                        olrp[k, a, m] = 1.
                        lrp_opt_thr[k, a, m] = np.nan
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'scores': scores,
            'olrp_loc': olrp_loc,
            'olrp_fp': olrp_fp,
            'olrp_fn': olrp_fn,
            'olrp': olrp,
            'lrp_opt_thr': lrp_opt_thr,
        }
        toc = time.time()
        self.log('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter
        setting
        '''
        def _summarize(ap=1,
                       iouThr=None,
                       areaRng='all',
                       maxDets=100,
                       lrp_type=None):
            p = self.params
            iStr = '{:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'  # noqa: E501
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [
                i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng
            ]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
                if len(s[s > -1]) == 0:
                    mean_s = -1
                else:
                    mean_s = np.mean(s[s > -1])
            elif ap == 0:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            else:
                # # dimension of LRP: [KxAxM]
                # Person 0, Broccoli 50
                if lrp_type == 'oLRP':
                    s = self.eval['olrp'][:, aind, mind]
                    titleStr = 'Optimal LRP'
                    typeStr = '    '
                if lrp_type == 'oLRP_Localisation':
                    s = self.eval['olrp_loc'][:, aind, mind]
                    titleStr = 'Optimal LRP Loc'
                    typeStr = '    '
                if lrp_type == 'oLRP_false_positive':
                    s = self.eval['olrp_fp'][:, aind, mind]
                    titleStr = 'Optimal LRP FP'
                    typeStr = '    '
                if lrp_type == 'oLRP_false_negative':
                    s = self.eval['olrp_fn'][:, aind, mind]
                    titleStr = 'Optimal LRP FN'
                    typeStr = '    '
                if lrp_type == 'oLRP_thresholds':
                    s = self.eval['lrp_opt_thr'][:, aind, mind].squeeze(axis=1)
                    titleStr = '# Class-specific LRP-Optimal Thresholds # \n'
                    typeStr = '    '
                    # Floor by using 3 decimal digits
                    self.log(titleStr)
                    self.log(np.round(s - 0.5 * 10**(-3), 3))
                    return s
            idx = (~np.isnan(s))
            s = s[idx]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            self.log(
                iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets,
                            mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((19, ))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1,
                                  iouThr=.75,
                                  maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1,
                                  areaRng='small',
                                  maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1,
                                  areaRng='medium',
                                  maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1,
                                  areaRng='large',
                                  maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0,
                                  areaRng='small',
                                  maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0,
                                   areaRng='medium',
                                   maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0,
                                   areaRng='large',
                                   maxDets=self.params.maxDets[2])
            stats[12] = _summarize(-1,
                                   iouThr=.5,
                                   areaRng='all',
                                   maxDets=self.params.maxDets[2],
                                   lrp_type='oLRP')
            stats[13] = _summarize(-1,
                                   iouThr=.5,
                                   areaRng='all',
                                   maxDets=self.params.maxDets[2],
                                   lrp_type='oLRP_Localisation')
            stats[14] = _summarize(-1,
                                   iouThr=.5,
                                   areaRng='all',
                                   maxDets=self.params.maxDets[2],
                                   lrp_type='oLRP_false_positive')
            stats[15] = _summarize(-1,
                                   iouThr=.5,
                                   areaRng='all',
                                   maxDets=self.params.maxDets[2],
                                   lrp_type='oLRP_false_negative')
            stats[16] = _summarize(-1,
                                   iouThr=.5,
                                   areaRng='small',
                                   maxDets=self.params.maxDets[2],
                                   lrp_type='oLRP')
            stats[17] = _summarize(-1,
                                   iouThr=.5,
                                   areaRng='medium',
                                   maxDets=self.params.maxDets[2],
                                   lrp_type='oLRP')
            stats[18] = _summarize(-1,
                                   iouThr=.5,
                                   areaRng='large',
                                   maxDets=self.params.maxDets[2],
                                   lrp_type='oLRP')
            if self.params.lrp_size_details:
                stats_lrp_size = np.zeros((9, ))
                stats_lrp_size[0] = _summarize(-1,
                                       iouThr=.5,
                                       areaRng='small',
                                       maxDets=self.params.maxDets[2],
                                       lrp_type='oLRP_Localisation')
                stats_lrp_size[1] = _summarize(-1,
                                       iouThr=.5,
                                       areaRng='medium',
                                       maxDets=self.params.maxDets[2],
                                       lrp_type='oLRP_Localisation')
                stats_lrp_size[2] = _summarize(-1,
                                       iouThr=.5,
                                       areaRng='large',
                                       maxDets=self.params.maxDets[2],
                                       lrp_type='oLRP_Localisation')
                stats_lrp_size[3] = _summarize(-1,
                                       iouThr=.5,
                                       areaRng='small',
                                       maxDets=self.params.maxDets[2],
                                       lrp_type='oLRP_false_positive')
                stats_lrp_size[4] = _summarize(-1,
                                       iouThr=.5,
                                       areaRng='medium',
                                       maxDets=self.params.maxDets[2],
                                       lrp_type='oLRP_false_positive')
                stats_lrp_size[5] = _summarize(-1,
                                       iouThr=.5,
                                       areaRng='large',
                                       maxDets=self.params.maxDets[2],
                                       lrp_type='oLRP_false_positive')
                stats_lrp_size[6] = _summarize(-1,
                                       iouThr=.5,
                                       areaRng='small',
                                       maxDets=self.params.maxDets[2],
                                       lrp_type='oLRP_false_negative')
                stats_lrp_size[7] = _summarize(-1,
                                       iouThr=.5,
                                       areaRng='medium',
                                       maxDets=self.params.maxDets[2],
                                       lrp_type='oLRP_false_negative')
                stats_lrp_size[8] = _summarize(-1,
                                       iouThr=.5,
                                       areaRng='large',
                                       maxDets=self.params.maxDets[2],
                                       lrp_type='oLRP_false_negative')
            _summarize(-1,
                       iouThr=.5,
                       areaRng='all',
                       maxDets=self.params.maxDets[2],
                       lrp_type='oLRP_thresholds')
            return stats

        def _summarizeKps():
            stats = np.zeros((16, ))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            stats[10] = _summarize(-1,
                                   maxDets=20,
                                   iouThr=.5,
                                   areaRng='all',
                                   lrp_type='oLRP')
            stats[11] = _summarize(-1,
                                   maxDets=20,
                                   iouThr=.5,
                                   areaRng='all',
                                   lrp_type='oLRP_Localisation')
            stats[12] = _summarize(-1,
                                   maxDets=20,
                                   iouThr=.5,
                                   areaRng='all',
                                   lrp_type='oLRP_false_positive')
            stats[13] = _summarize(-1,
                                   maxDets=20,
                                   iouThr=.5,
                                   areaRng='all',
                                   lrp_type='oLRP_false_negative')
            stats[14] = _summarize(-1,
                                   maxDets=20,
                                   iouThr=.5,
                                   areaRng='medium',
                                   lrp_type='oLRP')
            stats[15] = _summarize(-1,
                                   maxDets=20,
                                   iouThr=.5,
                                   areaRng='large',
                                   lrp_type='oLRP')
            if self.params.lrp_size_details:
                stats_lrp_size = np.zeros((6, ))
                stats_lrp_size[0] = _summarize(-1,
                                       maxDets=20,
                                       iouThr=.5,
                                       areaRng='medium',
                                       lrp_type='oLRP_Localisation')
                stats_lrp_size[1] = _summarize(-1,
                                       maxDets=20,
                                       iouThr=.5,
                                       areaRng='large',
                                       lrp_type='oLRP_Localisation')
                stats_lrp_size[2] = _summarize(-1,
                                       maxDets=20,
                                       iouThr=.5,
                                       areaRng='medium',
                                       lrp_type='oLRP_false_positive')
                stats_lrp_size[3] = _summarize(-1,
                                       maxDets=20,
                                       iouThr=.5,
                                       areaRng='large',
                                       lrp_type='oLRP_false_positive')
                stats_lrp_size[4] = _summarize(-1,
                                       maxDets=20,
                                       iouThr=.5,
                                       areaRng='medium',
                                       lrp_type='oLRP_false_negative')
                stats_lrp_size[5] = _summarize(-1,
                                       maxDets=20,
                                       iouThr=.5,
                                       areaRng='large',
                                       lrp_type='oLRP_false_negative')
            _summarize(-1,
                       iouThr=.5,
                       areaRng='all',
                       maxDets=20,
                       lrp_type='oLRP_thresholds')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()


class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly
        # larger than the true value
        self.iouThrs = np.linspace(.5,
                                   0.95,
                                   int(np.round((0.95 - .5) / .05)) + 1,
                                   endpoint=True)
        self.recThrs = np.linspace(.0,
                                   1.00,
                                   int(np.round((1.00 - .0) / .01)) + 1,
                                   endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2],
                        [96**2, 1e5**2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly
        # larger than the true value
        self.iouThrs = np.linspace(.5,
                                   0.95,
                                   int(np.round((0.95 - .5) / .05)) + 1,
                                   endpoint=True)
        self.recThrs = np.linspace(.0,
                                   1.00,
                                   int(np.round((1.00 - .0) / .01)) + 1,
                                   endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0**2, 1e5**2], [32**2, 96**2], [96**2, 1e5**2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0

    def __init__(self, iouType='segm', lrp_size_details = False):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        self.lrp_size_details = lrp_size_details
        # useSegm is deprecated
        self.useSegm = None
