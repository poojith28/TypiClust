# This file is slightly modified from a code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

from .Sampling import Sampling, CoreSetMIPSampling, AdversarySampler
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)


def _default_sampling_metadata(strategy_name, active_set=None):
    selected_count = 0 if active_set is None else int(len(active_set))
    return {
        'strategy': strategy_name,
        'selection_mode': strategy_name,
        'boundary_variant': 'not_applicable',
        'uncertainty_mode': 'not_applicable',
        'uncertainty_active': False,
        'coverage_fraction_before': 'not_applicable',
        'coverage_fraction_after': 'not_applicable',
        'components': 'not_applicable',
        'largest_component_fraction': 'not_applicable',
        'selected_count': selected_count,
    }

class ActiveLearning:
    """
    Implements standard active learning methods.
    """

    def __init__(self, dataObj, cfg):
        self.dataObj = dataObj
        self.sampler = Sampling(dataObj=dataObj,cfg=cfg)
        self.cfg = cfg
        self.latest_sampling_metadata = {}
        
    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None):
        """
        Sample from uSet using cfg.ACTIVE_LEARNING.SAMPLING_FN.

        INPUT
        ------
        clf_model: Reference of task classifier model class [Typically VGG]

        supportingModels: List of models which are used for sampling process.

        OUTPUT
        -------
        Returns activeSet, uSet
        """
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE > 0, "Expected a positive budgetSize"
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE < len(uSet), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}"\
        .format(len(uSet), self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)
        self.latest_sampling_metadata = {}

        if self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "random":

            activeSet, uSet = self.sampler.random(uSet=uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)
            self.latest_sampling_metadata = _default_sampling_metadata('random_policy', activeSet)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "uncertainty":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.uncertainty(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)
            self.latest_sampling_metadata = {
                **_default_sampling_metadata('uncertainty_policy', activeSet),
                'uncertainty_mode': 'least_confidence',
                'uncertainty_active': True,
            }
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "entropy":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.entropy(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)
            self.latest_sampling_metadata = {
                **_default_sampling_metadata('uncertainty_policy', activeSet),
                'uncertainty_mode': 'entropy',
                'uncertainty_active': True,
            }
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "margin":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.margin(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)
            self.latest_sampling_metadata = {
                **_default_sampling_metadata('uncertainty_policy', activeSet),
                'uncertainty_mode': 'margin',
                'uncertainty_active': True,
            }

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "coreset":
            waslatent = clf_model.penultimate_active
            wastrain = clf_model.training
            clf_model.penultimate_active = True
            # if self.cfg.TRAIN.DATASET == "IMAGENET":
            #     clf_model.cuda(0)
            clf_model.eval()
            coreSetSampler = CoreSetMIPSampling(cfg=self.cfg, dataObj=self.dataObj)
            activeSet, uSet = coreSetSampler.query(lSet=lSet, uSet=uSet, clf_model=clf_model, dataset=trainDataset)
            
            clf_model.penultimate_active = waslatent
            clf_model.train(wastrain)
            self.latest_sampling_metadata = _default_sampling_metadata('coreset_policy', activeSet)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("typiclust"):
            from .typiclust import TypiClust
            is_scan = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.endswith('dc')
            tpc = TypiClust(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, is_scan=is_scan)
            activeSet, uSet = tpc.select_samples()
            self.latest_sampling_metadata = _default_sampling_metadata('typiclust_policy', activeSet)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["prob_cover", 'probcover']:
            from .prob_cover import ProbCover
            probcov = ProbCover(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
            activeSet, uSet = probcov.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["id_prob_cover", "idprobcover"]:
            from .IDprocover import IDProbCover
            idpc = IDProbCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ALPHA', 1.0)),
                mode=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_MODE', 'high_id_more_centers')),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_CACHE_ROOT', './idprobcover_cache') or './idprobcover_cache'),
                k_id=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_ID', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_KNN', 50)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = idpc.select_samples()
            self.latest_sampling_metadata = getattr(idpc, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["idprobcover_tiebreak_min_id", "idprobcover_minid_tiebreak"]:
            from .idpc_tiebreak import IDProbCoverMinIDTieBreak
            idpc = IDProbCoverMinIDTieBreak(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ALPHA', 1.0)),
                mode=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_MODE', 'high_id_more_centers')),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_CACHE_ROOT', './idprobcover_cache') or './idprobcover_cache'),
                k_id=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_ID', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_KNN', 50)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = idpc.select_samples()
            self.latest_sampling_metadata = getattr(idpc, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["idprobcover_tiebreak_random", "idprobcover_random_tiebreak"]:
            from .idpc_tiebreak import IDProbCoverRandomTieBreak
            idpc = IDProbCoverRandomTieBreak(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ALPHA', 1.0)),
                mode=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_MODE', 'high_id_more_centers')),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_CACHE_ROOT', './idprobcover_cache') or './idprobcover_cache'),
                k_id=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_ID', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_KNN', 50)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = idpc.select_samples()
            self.latest_sampling_metadata = getattr(idpc, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["idprobcover_tiebreak_first_max", "idprobcover_firstmax_tiebreak"]:
            from .idpc_tiebreak import IDProbCoverFirstMaxTieBreak
            idpc = IDProbCoverFirstMaxTieBreak(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ALPHA', 1.0)),
                mode=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_MODE', 'high_id_more_centers')),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_CACHE_ROOT', './idprobcover_cache') or './idprobcover_cache'),
                k_id=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_ID', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_KNN', 50)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = idpc.select_samples()
            self.latest_sampling_metadata = getattr(idpc, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["knn_distance_cover", "adaptive_knn_distance_cover"]:
            from .adaptive_cover import KnnDistanceCover
            sampler = KnnDistanceCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_ALPHA', 1.0)),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_CACHE_ROOT', './adaptive_cover_cache') or './adaptive_cover_cache'),
                k_signal=int(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_K_SIGNAL', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_K_KNN', 50)),
                eps=float(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_EPS', 1e-8)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["density_cover", "adaptive_density_cover"]:
            from .adaptive_cover import DensityCover
            sampler = DensityCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_ALPHA', 1.0)),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_CACHE_ROOT', './adaptive_cover_cache') or './adaptive_cover_cache'),
                k_signal=int(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_K_SIGNAL', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_K_KNN', 50)),
                eps=float(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_EPS', 1e-8)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["distance_variance_cover", "adaptive_distance_variance_cover"]:
            from .adaptive_cover import DistanceVarianceCover
            sampler = DistanceVarianceCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_ALPHA', 1.0)),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_CACHE_ROOT', './adaptive_cover_cache') or './adaptive_cover_cache'),
                k_signal=int(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_K_SIGNAL', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_K_KNN', 50)),
                eps=float(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_EPS', 1e-8)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["distance_cv_cover", "adaptive_distance_cv_cover"]:
            from .adaptive_cover import DistanceCVCover
            sampler = DistanceCVCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_ALPHA', 1.0)),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_CACHE_ROOT', './adaptive_cover_cache') or './adaptive_cover_cache'),
                k_signal=int(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_K_SIGNAL', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_K_KNN', 50)),
                eps=float(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_EPS', 1e-8)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["geometry_auto_research", "geoar"]:
            from .geometry_auto_research import GeometryAutoResearch
            geoar = GeometryAutoResearch(
                self.cfg,
                lSet,
                uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                clf_model=clf_model,
                trainDataset=trainDataset,
                dataObj=self.dataObj,
            )
            activeSet, uSet = geoar.select_samples()
            self.latest_sampling_metadata = getattr(geoar, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["dcom"]:
            from .DCoM import DCoM
            dcom = DCoM(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        max_delta=self.cfg.ACTIVE_LEARNING.MAX_DELTA,
                        lSet_deltas=self.cfg.ACTIVE_LEARNING.DELTA_LST)
            activeSet, uSet = dcom.select_samples(clf_model, trainDataset, self.dataObj)
            self.latest_sampling_metadata = _default_sampling_metadata('dcom_policy', activeSet)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "dbal" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "DBAL":
            activeSet, uSet = self.sampler.dbal(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, \
                uSet=uSet, clf_model=clf_model,dataset=trainDataset)
            self.latest_sampling_metadata = _default_sampling_metadata('dbal_policy', activeSet)
            
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "bald" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "BALD":
            activeSet, uSet = self.sampler.bald(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, clf_model=clf_model, dataset=trainDataset)
            self.latest_sampling_metadata = _default_sampling_metadata('bald_policy', activeSet)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "ensemble_var_R":
            activeSet, uSet = self.sampler.ensemble_var_R(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, clf_models=supportingModels, dataset=trainDataset)
            self.latest_sampling_metadata = _default_sampling_metadata('ensemble_policy', activeSet)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "vaal":
            adv_sampler = AdversarySampler(cfg=self.cfg, dataObj=self.dataObj)

            # Train VAE and discriminator first
            vae, disc, uSet_loader = adv_sampler.vaal_perform_training(lSet=lSet, uSet=uSet, dataset=trainDataset)

            # Do active sampling
            activeSet, uSet = adv_sampler.sample_for_labeling(vae=vae, discriminator=disc, \
                                unlabeled_dataloader=uSet_loader, uSet=uSet)
            self.latest_sampling_metadata = _default_sampling_metadata('vaal_policy', activeSet)
        else:
            print(f"{self.cfg.ACTIVE_LEARNING.SAMPLING_FN} is either not implemented or there is some spelling mistake.")
            raise NotImplementedError

        return activeSet, uSet
        
