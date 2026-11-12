import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from torch._dynamo import OptimizedModule


from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from sklearn.model_selection import KFold
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


class nnUNetTrainer(object):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        # From https://grugbrain.dev/. Worth a read ya big brains ;-)

        # apex predator of grug is complexity
        # complexity bad
        # say again:
        # complexity very bad
        # you say now:
        # complexity very, very bad
        # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
        # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
        # one day code base understandable and grug can get work done, everything good!
        # next day impossible: complexity demon spirit has entered code and very dangerous situation!

        # OK OK I am guilty. But I tried. http://tiny.cc/gzgwuz

        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()

        self.device = device

        # print what device we are using
        if self.is_ddp:  # implicitly it's clear that we use cuda in this case
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
                  f"{dist.get_world_size()}."
                  f"Setting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
                self.device = torch.device(type='cuda', index=0)
            print(f"Using device: {self.device}")

        # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
        # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
        # need. So let's save the init args
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]

        ###  Saving all the init args into class variables for later access
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.configuration_name = configuration
        self.dataset_json = dataset_json
        self.fold = fold
        self.unpack_dataset = unpack_dataset

        ### Setting all the folder names. We need to make sure things don't crash in case we are just running
        # inference and some of the folders may not be defined!
        self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
            if nnUNet_preprocessed is not None else None
        self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                       self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
            if nnUNet_results is not None else None
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.configuration_manager.data_identifier)
        # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
        # be a different configuration in the same plans
        # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
        # "previous_stage" and "next_stage"). Otherwise it won't work!
        self.is_cascaded = self.configuration_manager.previous_stage_name is not None
        self.folder_with_segs_from_previous_stage = \
            join(nnUNet_results, self.plans_manager.dataset_name,
                 self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
                 self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
                if self.is_cascaded else None

        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-4
        self.weight_decay = 3e-6
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.current_epoch = 0

        ### Dealing with labels/regions
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
        # needed for predictions. We do sigmoid in case of (overlapping) regions

        self.num_input_channels = None  # -> self.initialize()
        self.network = None  # -> self._get_network()
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        self.loss = None  # -> self.initialize

        ### Simple logging. Don't take that away from me!
        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging
        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.logger = nnUNetLogger()

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start

        ### initializing stuff for remembering things and such
        self._best_ema = None

        ### inference things
        self.inference_allowed_mirroring_axes = None  # this variable is set in
        # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints

        ### checkpoint saving stuff
        self.save_every = 50
        self.disable_checkpointing = False

        ## DDP batch size and oversampling can differ between workers and needs adaptation
        # we need to change the batch size in DDP because we don't use any of those distributed samplers
        self._set_batch_size_and_oversample()

        self.was_initialized = False

        self.print_to_log_file("\n#######################################################################\n"
                               "Please cite the following paper when using nnU-Net:\n"
                               "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
                               "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
                               "Nature methods, 18(2), 203-211.\n"
                               "#######################################################################\n",
                               also_print_to_console=True, add_timestamp=False)

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                           self.configuration_manager,
                                                           self.num_input_channels,
                                                           enable_deep_supervision=True).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Compiling network...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def _do_i_compile(self):
        return ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))

    def _save_debug_information(self):
        # saving some debug information
        if self.local_rank == 0:
            dct = {}
            for k in self.__dir__():
                if not k.startswith("__"):
                    if not callable(getattr(self, k)) or k in ['loss', ]:
                        dct[k] = str(getattr(self, k))
                    elif k in ['network', ]:
                        dct[k] = str(getattr(self, k).__class__.__name__)
                    else:
                        pass
                if k in ['dataloader_train', 'dataloader_val']:
                    if hasattr(getattr(self, k), 'generator'):
                        dct[k + '.generator'] = str(getattr(self, k).generator)
                    if hasattr(getattr(self, k), 'num_processes'):
                        dct[k + '.num_processes'] = str(getattr(self, k).num_processes)
                    if hasattr(getattr(self, k), 'transform'):
                        dct[k + '.transform'] = str(getattr(self, k).transform)
            import subprocess
            hostname = subprocess.getoutput(['hostname'])
            dct['hostname'] = hostname
            torch_version = torch.__version__
            if self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name()
                dct['gpu_name'] = gpu_name
                cudnn_version = torch.backends.cudnn.version()
            else:
                cudnn_version = 'None'
            dct['device'] = str(self.device)
            dct['torch_version'] = torch_version
            dct['cudnn_version'] = cudnn_version
            save_json(dct, join(self.output_folder, "debug.json"))

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        his is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        but don't bug us with your bugs arising from fiddling with this :-P
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)

        """
        return get_network_from_plans(plans_manager, dataset_json, configuration_manager,
                                      num_input_channels, deep_supervision=enable_deep_supervision)

    def _get_deep_supervision_scales(self):
        deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]
        return deep_supervision_scales

    def _set_batch_size_and_oversample(self):
        if not self.is_ddp:
            # set batch size to what the plan says, leave oversample untouched
            self.batch_size = self.configuration_manager.batch_size
        else:
            # batch size is distributed over DDP workers and we need to change oversample_percent for each worker
            batch_sizes = []
            oversample_percents = []

            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            global_batch_size = self.configuration_manager.batch_size
            assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of ' \
                                                    'GPUs... Duh.'

            batch_size_per_GPU = np.ceil(global_batch_size / world_size).astype(int)

            for rank in range(world_size):
                if (rank + 1) * batch_size_per_GPU > global_batch_size:
                    batch_size = batch_size_per_GPU - ((rank + 1) * batch_size_per_GPU - global_batch_size)
                else:
                    batch_size = batch_size_per_GPU

                batch_sizes.append(batch_size)

                sample_id_low = 0 if len(batch_sizes) == 0 else np.sum(batch_sizes[:-1])
                sample_id_high = np.sum(batch_sizes)

                if sample_id_high / global_batch_size < (1 - self.oversample_foreground_percent):
                    oversample_percents.append(0.0)
                elif sample_id_low / global_batch_size > (1 - self.oversample_foreground_percent):
                    oversample_percents.append(1.0)
                else:
                    percent_covered_by_this_rank = sample_id_high / global_batch_size - sample_id_low / global_batch_size
                    oversample_percent_here = 1 - (((1 - self.oversample_foreground_percent) -
                                                    sample_id_low / global_batch_size) / percent_covered_by_this_rank)
                    oversample_percents.append(oversample_percent_here)

            print("worker", my_rank, "oversample", oversample_percents[my_rank])
            print("worker", my_rank, "batch_size", batch_sizes[my_rank])
            # self.print_to_log_file("worker", my_rank, "oversample", oversample_percents[my_rank])
            # self.print_to_log_file("worker", my_rank, "batch_size", batch_sizes[my_rank])

            self.batch_size = batch_sizes[my_rank]
            self.oversample_foreground_percent = oversample_percents[my_rank]

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = {
                    'x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                }
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnunet for now)
        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            *rotation_for_DA.values(),
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = ("%s:" % dt_object, *args)

            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)

    def print_plans(self):
        if self.local_rank == 0:
            dct = deepcopy(self.plans_manager.plans)
            del dct['configurations']
            self.print_to_log_file(f"\nThis is the configuration used by this "
                                   f"training:\nConfiguration name: {self.configuration_name}\n",
                                   self.configuration_manager, '\n', add_timestamp=False)
            self.print_to_log_file('These are the global plan.json settings:\n', dct, '\n', add_timestamp=False)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def plot_network_architecture(self):
        if self._do_i_compile():
            self.print_to_log_file("Unable to plot network architecture: nnUNet_compile is enabled!")
            return

        if self.local_rank == 0:
            try:

                import hiddenlayer as hl
                g = hl.build_graph(self.network,
                                   torch.rand((1, self.num_input_channels,
                                               *self.configuration_manager.patch_size),
                                              device=self.device),
                                   transforms=None)
                g.save(join(self.output_folder, "network_architecture.pdf"))
                del g
            except Exception as e:
                self.print_to_log_file("Unable to plot network architecture:")
                self.print_to_log_file(e)

                # self.print_to_log_file("\n")
            finally:
                empty_cache(self.device)

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            # val_keys = tr_keys
            # 閺夆晜鐟﹀Σ鎼佸礂閵娾晛鍔ラ悹渚囧幘缁矂寮懜闈涚憦婵炴潙顑堥惁顖炴⒖閸℃瑦绾柟鎭掑劚缂嶅顨ュ畝鍐闂?[婵炲备鍓濆﹢涔氶柤鏉垮缁楁牠宕氶幒鎴濈€诲ù?0濞戞搩浜滈崺宀€鎷嬮鐘电煀闂傚棗妫楃紞瀣▔?            # tr_keys = ['image_08702', 'image_08901', 'image_01602', 'image_07602', 'image_03402', 'image_05801', 'image_07301', 'image_10802', 'image_04602', 'image_04102', 'image_10202', 'image_07402', 'image_05201', 'image_09202', 'image_06502', 'image_08701', 'image_06501', 'image_08002', 'image_01301', 'image_04302', 'image_06801', 'image_03901', 'image_03501', 'image_09302', 'image_03801', 'image_01202', 'image_09702', 'image_08001', 'image_02801', 'image_10501', 'image_09402', 'image_09201', 'image_09902', 'image_01401', 'image_02301', 'image_02702', 'image_05901', 'image_03201', 'image_08302', 'image_06001', 'image_08802', 'image_00601', 'image_07502', 'image_06401', 'image_00201', 'image_07001', 'image_00701', 'image_04901', 'image_08401', 'image_05702', 'image_07801', 'image_08102', 'image_04802', 'image_05101', 'image_10801', 'image_00202', 'image_01601', 'image_01302', 'image_07802', 'image_01402', 'image_06101', 'image_10502', 'image_07501', 'image_03002', 'image_08501', 'image_05902', 'image_06301', 'image_05301', 'image_00802', 'image_05502', 'image_04501', 'image_00602', 'image_06302', 'image_06402', 'image_02001', 'image_02502', 'image_04702', 'image_10102', 'image_03301', 'image_02102', 'image_02701', 'image_04801', 'image_03202', 'image_02101', 'image_06002', 'image_01102', 'image_05501', 'image_01101', 'image_02002', 'image_04701', 'image_10001', 'image_08202', 'image_04002', 'image_06602', 'image_02802', 'image_05202', 'image_04301', 'image_04502', 'image_10701', 'image_00901', 'image_02401', 'image_07002', 'image_09901', 'image_02501', 'image_06802', 'image_06102', 'image_07101', 'image_09101', 'image_03001', 'image_03102', 'image_09502', 'image_07401', 'image_08101', 'image_07601', 'image_02202', 'image_05602', 'image_07102', 'image_04101', 'image_03302', 'image_00902', 'image_08502', 'image_06201', 'image_01201', 'image_09701', 'image_04001', 'image_08402', 'image_09301', 'image_03401', 'image_07702', 'image_00401', 'image_08301', 'image_08902', 'image_00101', 'image_03802', 'image_05601', 'image_03902', 'image_05701', 'image_02201', 'image_07701', 'image_03502', 'image_09102', 'image_00402', 'image_09401', 'image_10702', 'image_02302', 'image_08201', 'image_10201', 'image_10101', 'image_00702', 'image_07302', 'image_03701', 'image_09801', 'image_03702', 'image_04601', 'image_00801', 'image_03101', 'image_09501', 'image_07201', 'image_00102', 'image_02402', 'image_04902', 'image_00301', 'image_00302', 'image_05102', 'image_08801', 'image_05802', 'image_10002', 'image_07202', 'image_05302', 'image_10601', 'image_06601', 'image_10602', 'image_06202', 'image_09802', 'image_017', 'image_019', 'image_036', 'image_090', 'image_044', 'image_050', 'image_054', 'image_067', 'image_079', 'image_086', 'image_005', 'image_018', 'image_026', 'image_029', 'image_096']
            # val_keys = ['image_005', 'image_010', 'image_015', 'image_018', 'image_026', 'image_029', 'image_042', 'image_069', 'image_096', 'image_103', 'image_104']

            # 閻庣數鎳撶敮顐ｆ叏鐎ｎ剚鈷曞〒姘儑濞堟唨owres閺夆晜绋栭、鎴﹀礆閸℃顥?            # tr_keys = ["image_001", "image_002", "image_004", "image_007","image_009","image_011","image_012","image_013","image_014","image_016","image_021",
            # "image_022","image_023","image_024","image_025","image_027","image_030","image_031","image_032","image_033","image_034","image_037","image_038","image_039",
            # "image_040","image_041","image_043","image_046","image_048","image_049","image_051","image_052","image_053","image_055","image_056","image_057",
            # "image_058","image_059","image_061","image_062","image_064","image_065","image_066","image_068","image_070","image_071",
            # "image_072","image_073","image_075","image_076","image_077","image_080","image_081","image_083","image_085",
            # "image_089","image_091","image_092", "image_093","image_094","image_095","image_097","image_098", "image_099","image_101",
            # "image_102","image_106","image_107","image_108", "image_003","image_006","image_008","image_020","image_028","image_035",
            # "image_045","image_047","image_060","image_063","image_074","image_078","image_082","image_084","image_087","image_088",
            # "image_100","image_105", 'image_017', 'image_019', 'image_036', 'image_090', 'image_044', 'image_050', 'image_054', 'image_067', 'image_079', 'image_086']
            # val_keys = ['image_005', 'image_010', 'image_015', 'image_018', 'image_026', 'image_029', 'image_042', 'image_069', 'image_096', 'image_103', 'image_104']

            # ================= MAS闁轰胶澧楀畵渚€姊?=======================
            # 閺夆晜鐟﹀Σ鎼佸箰閹达腹鍋?0%濞达絾绮堢拹鐔访圭€ｎ厾妲搁梻?            # tr_keys = ['0000706877', '0000858081', '0000626470', '0000570636', '0000591801', '0000874742', '0000425994', '0000792474', '0000862853', '0000925566', '0000157009', '0000776724', '0001016478', '0000352025', '0000151253', '0000933988', '0000111371', '0000876881', '0000007212', '0001017295', '0000469491', '0001019586', '0001017612', '0000178011', '0000009426', '0000974965', '0000894607', '0000981387', '0000420105', '0000000692', '0000931699', '0000046981', '0000900932', '0000038985', '0000348024', '0000230028', '0000233748', '0000916594', '0000971138', '0000258124', '0000072306', '0000384595', '0000786529', '0000097259', '0000391043', '0000638451', '0000612982', '0000025745', '0000042693', '0000466994', '0000963377', '0000264877', '0000667324', '0000868666', '0000686835', '0000095216', '0000420571', '0000966323', '0000861405', '0000965795', '0000972912', '0000163718', '0000924122', '0000979454', '0000987587', '0000977646', '0000925737', '0000990926', '0000306571', '0000959870', '0000927108', '0000454967', '0000963538', '0000231737', '0000098621', '0000447651', '0000410329', '0000231397', '0000859304', '0000724089', '0000954259', '0000996543', '0000951772', '0001015136', '0000986040', '0000065253', '0000510480', '0000956102', '0000872440', '0000996658']
            # val_keys = ['0000154597', '0000825658', '0000864918', '0000899179', '0000862266', '0000109572', '0000118857', '0000683270', '0001014213', '0000962629']
            # 閺夆晜鐟﹀Σ鍛婃媴鐠恒劍鏆?0%閺夆晜绋栭、鎴︽嚂濮樺崬濡介梻鍐煐椤斿矂鎯冮崚顩弉etune
            # tr_keys = ['0000683270', '0000962629', '0000862853', '0000151253', '0000111371', '0000469491', '0000009426', '0000894607', '0000981387', '0000000692', '0000038985', '0000230028', '0000384595', '0000097259', '0000963377', '0000868666', '0000965795', '0000154597', '0000825658', '0000864918', '0000899179', '0000862266', '0000109572', '0000118857', '0001014213', '0000706877', '0000858081', '0000626470', '0000570636', '0000591801']
            # val_keys = ['0000874742', '0000425994', '0000792474', '0000925566', '0000157009', '0000776724', '0001016478', '0000352025', '0000933988', '0000876881', '0000007212', '0001017295', '0001019586', '0001017612', '0000178011', '0000974965', '0000420105', '0000931699', '0000046981', '0000900932', '0000348024', '0000233748', '0000916594', '0000971138', '0000258124', '0000072306', '0000786529', '0000391043', '0000638451', '0000612982', '0000025745', '0000042693', '0000466994', '0000264877', '0000667324', '0000686835', '0000095216', '0000420571', '0000966323', '0000861405', '0000972912', '0000163718', '0000924122', '0000979454', '0000987587', '0000977646', '0000925737', '0000990926', '0000306571', '0000959870', '0000927108', '0000454967', '0000963538', '0000231737', '0000098621', '0000447651', '0000410329', '0000231397', '0000859304', '0000724089', '0000954259', '0000996543', '0000951772', '0001015136', '0000986040', '0000065253', '0000510480', '0000956102', '0000872440', '0000996658']
            # 閺夆晜鐟﹀Σ鍛婃媴鐠恒劍鏆?0%閺夆晜绋栭、鎴犲垝閸撗冧还闂傚啳鍩栭宀勬儍閸掝洀netune
            # ============================60闁?6=============================
            # tr_keys = ['0000862853_1', '0000151253_0', '0000862266_0', '0000009426_0', '0000154597_1', '0000683270_0',
            #            '0000858081_1', '0000230028_1', '0000591801_0', '0000109572_0', '0000000692_1', '0000469491_1',
            #            '0000858081_0',
            #            '0000384595_1', '0000981387_1', '0000825658_0', '0000111371_0', '0000111371_1',
            #            '0000038985_0',
            #            '0000862266_1', '0000868666_0', '0000899179_1', '0000626470_1', '0000962629_0', '0001014213_0',
            #            '0000000692_0',
            #            '0000862853_0', '0000151253_1', '0000118857_1', '0000570636_1', '0000963377_1', '0000962629_1',
            #            '0000963377_0',
            #            '0000591801_1', '0000706877_1', '0000899179_0', '0000864918_0', '0000009426_1', '0000097259_1',
            #            '0000965795_0',
            #            '0000683270_1', '0000864918_1', '0000706877', '0000965795_1', '0000230028_0', '0001014213_1',
            #            '0000154597_0',
            #            '0000894607_1', '0000894607_0', '0000981387_0', '0000570636_0', '0000109572_1', '0000626470_0',
            #            '0000825658_1',
            #            '0000384595_0', '0000097259_0', '0000868666_1', '0000469491_0', '0000118857_0']
            tr_keys = ['0000648594_1', '0000231397_1', '0000736360_1', '0000996658_1', '0000163718_1',
                       '0001019586_1', '0000924122_1', '0000956102_1', '0000562386_1', '0000931699_1',
                       '0000540498_1', '0000996543_1', '0000391043_1', '0000421192_1', '0000231737_1',
                       '0000861405_1', '0001017295_1', '0000264877_1', '0000233748_1', '0000954259_1',
                       '0000181566_1', '0000056863_1', '0000561064_1', '0000874742_1', '0000963377_1',
                       '0000951772_1', '0000776724_1', '0000354402_1', '0000916594_1', '0000072306_1',
                       '0000099489_1', '0000706877_1', '0000042693_1', '0000108941_1', '0001017612_1',
                       '0000732171_1', '0000858081_1', '0000126611_1', '0000876881_1', '0000971138_1',
                       '0000825658_1', '0000742142_1', '0000719066_1', '0000744155_1', '0000745235_1',
                       '0000872440_1', '0000667324_1', '0000410329_1', '0000040000_1', '0000510480_1',
                       '0000638451_1', '0000710433_1', '0000150573_1', '0000501060_1', '0001015136_1',
                       '0000218994_1', '0000178011_1', '0000208578_1', '0000591801_1', '0000245347_1',
                       # '0000966323_1', '0000862853_1', '0000706836_1', '0000715767_1', '0000447651_1',
                       # '0000118857_1', '0000539967_1', '0000724089_1', '0000666275_1', '0000611721_1',
                       # '0000535326_1', '0000602569_1', '0000000692_1', '0000258124_1', '0000612982_1',
                       # '0000899179_1', '0000348024_1', '0000979454_1', '0000336128_1', '0000986040_1',
                       # '0000736374_1', '0000925737_1', '0000306571_1', '0000925566_1', '0001016478_1',
                       # '0000686835_1', '0000425994_1', '0000900932_1', '0000626470_1', '0000111371_1',
                       # '0000987587_1', '0000509676_1', '0000609232_1', '0000697937_1', '0000098621_1', '0000712063_1',

                       # '0000666275_0', '0000231737_0', '0000258124_0', '0000391043_0', '0000996543_0',
                       # '0000535326_0', '0000715767_0', '0001016478_0', '0000126611_0', '0000706836_0',
                       # '0000899179_0', '0000742142_0', '0000986040_0', '0000591801_0', '0000609232_0',
                       # '0000732171_0', '0000648594_0', '0000736374_0', '0000925566_0', '0000996658_0',
                       # '0000719066_0', '0000931699_0', '0000118857_0', '0000108941_0', '0000686835_0',
                       # '0000916594_0', '0000072306_0', '0000954259_0', '0000098621_0', '0000924122_0',
                       # '0000874742_0', '0000979454_0', '0000667324_0', '0000421192_0', '0001017295_0',
                       # '0000306571_0', '0000638451_0', '0000150573_0', '0000540498_0', '0000825658_0',
                       # '0000056863_0', '0000218994_0', '0000561064_0', '0000208578_0', '0001015136_0',
                       # '0001017612_0', '0000744155_0', '0000712063_0', '0000231397_0', '0000510480_0',
                       # '0000956102_0', '0000163718_0', '0000872440_0', '0000501060_0', '0000562386_0',
                       # '0000602569_0', '0000000692_0', '0000539967_0', '0000348024_0', '0000925737_0',
                       # '0000410329_0', '0000178011_0', '0001019586_0', '0000858081_0', '0000040000_0',
                       # '0000987587_0', '0000612982_0', '0000966323_0', '0000776724_0', '0000900932_0',
                       # '0000181566_0', '0000861405_0', '0000447651_0', '0000963377_0', '0000745235_0',
                       # '0000706877_0', '0000099489_0', '0000233748_0', '0000042693_0', '0000425994_0',
                       # '0000611721_0', '0000971138_0', '0000264877_0', '0000336128_0', '0000111371_0',
                       # '0000876881_0', '0000245347_0', '0000951772_0', '0000697937_0', '0000710433_0',
                       # '0000724089_0', '0000862853_0', '0000626470_0', '0000736360_0', '0000354402_0', '0000509676_0'
                       ]
            val_keys = ['0000990926', '0000626322', '0000638155', '0000466994', '0000092333',
                        '0000217989', '0000420571', '0000100099', '0000065253', '0001014213',
                        '0000966323_1', '0000862853_1', '0000706836_1', '0000715767_1', '0000447651_1',
                        '0000118857_1', '0000539967_1', '0000724089_1', '0000666275_1', '0000611721_1',
                        '0000535326_1', '0000602569_1', '0000000692_1', '0000258124_1', '0000612982_1',
                        '0000899179_1', '0000348024_1', '0000979454_1', '0000336128_1', '0000986040_1',
                        '0000736374_1', '0000925737_1', '0000306571_1', '0000925566_1', '0001016478_1',
                        '0000686835_1', '0000425994_1', '0000900932_1', '0000626470_1', '0000111371_1',
                        '0000987587_1', '0000509676_1', '0000609232_1', '0000697937_1', '0000098621_1', '0000712063_1',

                        ]
            # val_keys = ['0000874742', '0000425994', '0000776724', '0001016478', '0000876881',
            #             '0001017295',
            #             '0001019586', '0001017612', '0000178011', '0000420105', '0000931699', '0000900932',
            #             '0000348024', '0000233748',
            #             '0000916594', '0000971138', '0000072306', '0000391043', '0000638451', '0000612982',
            #             '0000042693', '0000466994',
            #             '0000264877', '0000667324', '0000686835', '0000420571', '0000966323', '0000861405',
            #             '0000163718', '0000924122',
            #             '0000979454', '0000987587', '0000925737', '0000990926', '0000306571', '0000959870',
            #             '0000927108', '0000454967',
            #             '0000231737', '0000447651', '0000410329', '0000231397', '0000859304', '0000724089',
            #             '0000954259', '0000996543',
            #             '0001015136', '0000986040', '0000065253', '0000510480', '0000956102', '0000872440',
            #             '0000996658']

            # =======================MAS闁轰胶澧楀畵渚€姊块崱顓犵濞达絽妫欏Σ鎼佸储婵犳碍鐝熷ù婊冩缂嶅棛鎷归妸鈺佹闁?6濞?==============================================
            # 閺夆晜鐟﹀Σ鍛婃媴鐠恒劍鏆?0%閺夆晜绋栭、鎴︽嚂濮樺崬濡介梻鍐煐椤斿矂鎯冮崚顩弉etune
            # val_keys = ['0001019586', '0001017612', '0000178011', '0000931699', '0000900932', '0000348024', '0000233748', '0000916594', '0000971138', '0000258124', '0000072306', '0000391043', '0000638451', '0000612982', '0000042693', '0000466994', '0000264877', '0000667324', '0000686835', '0000420571', '0000966323', '0000861405', '0000163718', '0000924122', '0000979454', '0000987587', '0000925737', '0000990926', '0000306571', '0000231737', '0000098621', '0000447651', '0000410329', '0000231397', '0000724089', '0000954259', '0000996543', '0000951772', '0001015136', '0000986040', '0000065253', '0000510480', '0000956102', '0000872440', '0000996658']
            # tr_keys = ['0000862853', '0000111371', '0000000692', '0000963377', '0000825658', '0000899179', '0000118857', '0001014213', '0000706877', '0000858081', '0000626470', '0000591801', '0000874742', '0000425994', '0000925566', '0000776724', '0001016478', '0000876881', '0001017295']

            # ======================= MAS闁轰胶澧楀畵渚€姊块崱顓犵濞戞柨顑呮晶鐘绘儍?4濞?闁哄倹婢橀·鍐礉閻樺灚鐣?2濞?==================================
            # 86-20-0.65,
            # tr_keys = ['0000056863', '0000535326', '0000710433', '0000732171', '0000208578','0000118857',  '0000862853', '0000925566', '0000561064', '0000539967', '0001016478', '0000099489', '0000742142', '0000111371', '0001017295', '0000218994', '0001019586', '0001017612', '0000706836', '0000872440']
            # val_keys = ['0000178011', '0000666275', '0000000692', '0000931699', '0000562386', '0000900932', '0000648594', '0000638155', '0000697937', '0000348024', '0000540498', '0000233748', '0000501060', '0000916594', '0000971138', '0000258124', '0000072306', '0000391043', '0000092333', '0000638451', '0000612982', '0000126611', '0000042693', '0000040000', '0000466994', '0000825658', '0000963377', '0000181566', '0000264877', '0000667324', '0000686835', '0000100099', '0000420571', '0000602569', '0000966323', '0000736360', '0000861405', '0000719066', '0000163718', '0000924122', '0000611721', '0000979454', '0000987587', '0000609232', '0000744155', '0000421192', '0000336128', '0000626322', '0000925737', '0000712063', '0000990926', '0000715767', '0000899179', '0000509676', '0000306571', '0000108941', '0000736374', '0000231737', '0000098621', '0000447651', '0000410329', '0000354402', '0000231397', '0000724089', '0000954259', '0000996543', '0000951772', '0001015136', '0000986040', '0000065253', '0000150573', '0000510480', '0000956102', '0000996658', '0000245347',
            #             '0000876881', '0000776724', '0001014213', '0000706877', '0000858081', '0000626470', '0000591801', '0000745235', '0000874742', '0000217989', '0000425994',]    # 閺夆晜鐟﹀Σ鍛婄鎼淬値鍞茬紓浣稿暣濞夛附绋夐銏╂澓濠㈣埖鐗楅弻濠囧礉閻樺灚鐣?0濞?
            # ======================= MAS闁轰胶澧楀畵渚€姊块崱顓犵濞戞柨顑呮晶鐘绘儍?4濞?闁哄倹婢橀·鍐礉閻樺灚鐣?2濞?鐎甸偊鍠涢惃?=================================
            # # 60-46,0000056863缂傚倸鎼妵?            # tr_keys = ['0000591801', '0000745235', '0000874742', '0000217989', '0000535326', '0000710433',
            #            '0000732171', '0000208578', '0000118857', '0000862853','0000956102', '0000996658', '0000245347',
            #            '0000925566', '0000561064', '0000539967', '0001016478', '0000099489', '0000742142', '0000111371',
            #            '0000876881', '0000776724', '0001014213', '0000706877', '0000858081', '0000626470','0000425994',
            #            '0000336128', '0000626322',
            #            '0000925737', '0000712063', '0000990926', '0000715767', '0000899179', '0000509676',
            #            '0000306571', '0000108941', '0000736374', '0000231737', '0000098621', '0000447651',
            #            '0000410329', '0000354402', '0000231397', '0000724089', '0000954259', '0000996543'
            #            '0000951772', '0001015136', '0000986040', '0000065253', '0000150573', '0000510480',
            #            '0001017295', '0000218994', '0001019586', '0001017612', '0000706836', '0000872440']
            # val_keys = ['0000178011', '0000666275', '0000000692', '0000931699', '0000562386', '0000900932',
            #             '0000648594', '0000638155', '0000697937', '0000348024', '0000540498', '0000233748',
            #             '0000501060', '0000916594', '0000971138', '0000258124', '0000072306', '0000391043',
            #             '0000092333', '0000638451', '0000612982', '0000126611', '0000042693', '0000040000',
            #             '0000466994', '0000825658', '0000963377', '0000181566', '0000264877', '0000667324',
            #             '0000686835', '0000100099', '0000420571', '0000602569', '0000966323', '0000736360',
            #             '0000861405', '0000719066', '0000163718', '0000924122', '0000611721', '0000979454',
            #             '0000987587', '0000609232', '0000744155', '0000421192', ]  # 閺夆晜鐟﹀Σ鍛婄鎼淬値鍞茬紓浣稿暣濞夛附绋夐銏╂澓濠㈣埖鐗楅弻濠囧礉閻樺灚鐣?0濞?
            # ============================= MAS2闁轰胶澧楀畵渚€姊块崱顓犵缂備礁妫楅崹搴ㄥ礈閺屻儲鈻夋繛?鐎甸偊鍠涢惃?===========================
            # tr_keys = ['0000862853_1', '0000056863_1', '0001017612_0', '0000056863_0', '0001017295_0', '0000732171_0',
            # '0000218994_1', '0000111371_0', '0000925566_1', '0000111371_1', '0000561064_1', '0000208578_0', '0000742142_0',
            # '0001016478_0', '0000872440_0', '0001019586_0', '0001019586_1', '0000862853_0', '0000118857_1', '0000732171_1',
            # '0001017612_1', '0000710433_1', '0000925566_0', '0001017295_1', '0000539967_1', '0000710433_0', '0000535326_1',
            # '0000539967_0', '0000706836_1', '0001016478_1', '0000742142_1', '0000872440_1', '0000208578_1', '0000561064_0',
            # '0000706836_0', '0000099489_0', '0000218994_0', '0000099489_1', '0000535326_0', '0000118857_0']
            # val_keys = ['0000825658', '0000509676', '0000744155', '0001014213', '0000706877', '0000858081', '0000626470',
            # '0000591801', '0000745235', '0000874742', '0000217989', '0000425994', '0000776724', '0000876881', '0000178011',
            # '0000666275', '0000000692', '0000931699', '0000562386', '0000900932', '0000648594', '0000638155', '0000697937',
            # '0000348024', '0000540498', '0000233748', '0000501060', '0000916594', '0000971138', '0000072306', '0000391043',
            # '0000092333', '0000638451', '0000612982', '0000126611', '0000042693', '0000040000', '0000466994', '0000963377',
            # '0000181566', '0000264877', '0000667324', '0000686835', '0000100099', '0000420571', '0000602569', '0000966323',
            # '0000736360', '0000861405', '0000719066', '0000163718', '0000924122', '0000611721', '0000979454', '0000987587',
            # '0000609232', '0000421192', '0000626322', '0000925737', '0000712063', '0000990926', '0000715767', '0000306571',
            # '0000108941', '0000736374', '0000098621', '0000447651', '0000410329', '0000354402', '0000231397', '0000724089',
            # '0000954259', '0000996543', '0001015136', '0000986040', '0000065253', '0000150573', '0000510480', '0000956102',
            # '0000996658', '0000245347']

            # ============================= MAS2闁轰胶澧楀畵渚€姊块崱顓犵闁活亞鍠庨悿鍕媼椤撶姷鐭?闁艰鲸姘ㄩ崝宥夋⒓閼告鍞?==============================
            # tr_keys = ['0000862853', '0000899179', '0000510480', '0000951772', '0000732171', '0000954259', '0000233748', '0000208578', '0000996658', '0001017295', '0000056863', '0000126611', '0000924122', '0000118857', '0000745235', '0000900932', '0000724089', '0000072306', '0000181566', '0000501060', '0000706877', '0000744155', '0000742142', '0000150573', '0000602569', '0001015136', '0000391043', '0000348024', '0000447651', '0000354402', '0000336128', '0000040000', '0000264877', '0000591801', '0000539967', '0000245347', '0000963377', '0000987587', '0000231397', '0000666275', '0000562386', '0000178011', '0000561064', '0000425994', '0000410329', '0000638451', '0001017612', '0000535326', '0000697937', '0000825658', '0000231737', '0000956102', '0000540498', '0000736374', '0000986040', '0000258124', '0000648594', '0000966323', '0000916594', '0000609232', '0000925737', '0000996543', '0000686835', '0000776724', '0000626470', '0000306571', '0000667324', '0000706836', '0000712063', '0000509676', '0000421192', '0000736360', '0000108941', '0000218994', '0001016478', '0000861405', '0000876881', '0000715767', '0000719066', '0000971138', '0000111371', '0000710433', '0000612982', '0000098621', '0000611721', '0000925566', '0000872440', '0000979454', '0000000692', '0000858081', '0000874742', '0000042693', '0000099489', '0001019586', '0000931699', '0000163718']
            # val_keys = ['0000065253', '0000092333', '0000100099', '0000217989', '0000420571', '0000466994', '0000626322',
            #  '0000638155', '0000990926', '0001014213']

            # ============================ MAS2闁轰胶澧楀畵渚€姊?闁活亞鍠庨悿鍕媼椤撶姷鐭?缂侇喖澧介崑褔姊奸懜娈垮斀 =============================
            # tr_keys = ['0000648594_1', '0000231397_1', '0000666275_0', '0000736360_1', '0000231737_0', '0000996658_1', '0000163718_1', '0001019586_1', '0000924122_1', '0000956102_1', '0000562386_1', '0000258124_0', '0000391043_0', '0000931699_1', '0000540498_1', '0000996543_0', '0000996543_1', '0000535326_0', '0000391043_1', '0000715767_0', '0001016478_0', '0000126611_0', '0000706836_0', '0000899179_0', '0000421192_1', '0000231737_1', '0000742142_0', '0000986040_0', '0000861405_1', '0000591801_0', '0001017295_1', '0000609232_0', '0000264877_1', '0000732171_0', '0000233748_1', '0000954259_1', '0000181566_1', '0000056863_1', '0000561064_1', '0000874742_1', '0000963377_1', '0000648594_0', '0000736374_0', '0000951772_1', '0000925566_0', '0000996658_0', '0000776724_1', '0000719066_0', '0000931699_0', '0000354402_1', '0000916594_1', '0000118857_0', '0000072306_1', '0000108941_0', '0000099489_1', '0000706877_1', '0000686835_0', '0000916594_0', '0000072306_0', '0000954259_0', '0000098621_0', '0000042693_1', '0000108941_1', '0001017612_1', '0000924122_0', '0000732171_1', '0000858081_1', '0000874742_0', '0000126611_1', '0000979454_0', '0000876881_1', '0000971138_1', '0000825658_1', '0000667324_0', '0000742142_1', '0000719066_1', '0000421192_0', '0000744155_1', '0001017295_0', '0000306571_0', '0000745235_1', '0000638451_0', '0000872440_1', '0000667324_1', '0000150573_0', '0000540498_0', '0000410329_1', '0000040000_1', '0000510480_1', '0000825658_0', '0000638451_1', '0000056863_0', '0000710433_1', '0000218994_0', '0000561064_0', '0000150573_1', '0000501060_1', '0001015136_1', '0000218994_1', '0000208578_0', '0001015136_0', '0000178011_1', '0001017612_0', '0000208578_1', '0000591801_1', '0000744155_0', '0000245347_1', '0000712063_0', '0000966323_1', '0000862853_1', '0000231397_0', '0000510480_0', '0000706836_1', '0000715767_1', '0000956102_0', '0000447651_1', '0000163718_0', '0000118857_1', '0000539967_1', '0000872440_0', '0000501060_0', '0000562386_0', '0000602569_0', '0000724089_1', '0000666275_1', '0000000692_0', '0000611721_1', '0000535326_1', '0000539967_0', '0000602569_1', '0000348024_0', '0000925737_0', '0000410329_0', '0000178011_0', '0000000692_1', '0000258124_1', '0000612982_1', '0001019586_0', '0000899179_1', '0000348024_1', '0000858081_0', '0000040000_0', '0000987587_0', '0000612982_0', '0000979454_1', '0000336128_1', '0000986040_1', '0000736374_1', '0000925737_1', '0000966323_0', '0000306571_1', '0000776724_0', '0000900932_0', '0000925566_1', '0000181566_0', '0000861405_0', '0001016478_1', '0000447651_0', '0000963377_0', '0000686835_1', '0000745235_0', '0000706877_0', '0000099489_0', '0000233748_0', '0000425994_1', '0000042693_0', '0000425994_0', '0000900932_1', '0000611721_0', '0000971138_0', '0000264877_0', '0000336128_0', '0000111371_0', '0000626470_1', '0000876881_0', '0000245347_0', '0000111371_1', '0000951772_0', '0000697937_0', '0000987587_1', '0000710433_0', '0000509676_1', '0000609232_1', '0000724089_0', '0000862853_0', '0000697937_1', '0000626470_0', '0000736360_0', '0000098621_1', '0000354402_0', '0000712063_1', '0000509676_0']
            # val_keys = ['0000065253', '0000092333', '0000100099', '0000217989', '0000420571', '0000466994', '0000626322',
            #  '0000638155', '0000990926', '0001014213']

            # ===========================MAS2闁轰胶澧楀畵渚€姊块崱鏇犵憪閻犱緡鍘剧划宀勬晬鐏炶姤韬琖MC闁轰胶澧楀畵渚€姊块崱鏇犵憪鐎甸偊鍠涢惃?=============================
            # tr_keys = ['image_001', 'image_002', 'image_009', 'image_010', 'image_018', 'image_022', 'image_024', 'image_027', 'image_029', 'image_031', 'image_034', 'image_037', 'image_039', 'image_040', 'image_043', 'image_045', 'image_052', 'image_053', 'image_057', 'image_061', 'image_062', 'image_063', 'image_064', 'image_065', 'image_066', 'image_068', 'image_071', 'image_072', 'image_074', 'image_075', 'image_076', 'image_078']
            # val_keys = ['image_048', 'image_073', 'image_007', 'image_081', 'image_093', 'image_026', 'image_013', 'image_105', 'image_014', 'image_104', 'image_055', 'image_028', 'image_035', 'image_003', 'image_015', 'image_060', 'image_020', 'image_033', 'image_103', 'image_100', 'image_079', 'image_096', 'image_086', 'image_089', 'image_004', 'image_044', 'image_087', 'image_070', 'image_098', 'image_017', 'image_049', 'image_084', 'image_047', 'image_069', 'image_094', 'image_006', 'image_099', 'image_019', 'image_023', 'image_032', 'image_088', 'image_102', 'image_106', 'image_030', 'image_056', 'image_101', 'image_090', 'image_067', 'image_041', 'image_021', 'image_080', 'image_036', 'image_054', 'image_085', 'image_012', 'image_046', 'image_092', 'image_005', 'image_025', 'image_097', 'image_038', 'image_008', 'image_083', 'image_016', 'image_051', 'image_042', 'image_095', 'image_059', 'image_091', 'image_077', 'image_107', 'image_011', 'image_082', 'image_050', 'image_058', 'image_108']

            # =========================闁稿﹥鐟╅埀?=========================================================
            # tr_keys = ['0000056863', '0000535326', '0000710433', '0000732171', '0000208578','0000118857',  '0000862853',
            # '0000925566', '0000561064', '0000539967', '0001016478', '0000099489', '0000742142', '0000111371', '0001017295', '0000218994', '0001019586', '0001017612', '0000706836', '0000872440', '0000000692', '0000098621', '0000181566', '0000231737', '0000410329', '0000501060', '0000510480', '0000611721', '0000712063', '0000951772', '0000956102', '0000916594', '0000900932', '0000638155', '0000986040', '0000963377', '0000899179', '0000858081', '0000697937', '0000447651', '0000996543', '0000971138', '0000954259', '0000667324', '0000231397', '0000924122', '0000736374', '0000591801', '0000391043', '0000126611']
            # val_keys = ['0000178011', '0000666275', '0000931699', '0000562386', '0000648594', '0000348024', '0000540498', '0000233748', '0000258124', '0000072306', '0000092333', '0000638451', '0000612982', '0000042693', '0000040000', '0000466994', '0000825658', '0000264877', '0000686835', '0000100099', '0000420571', '0000602569', '0000966323', '0000736360', '0000861405', '0000719066', '0000163718', '0000979454', '0000987587', '0000609232', '0000744155', '0000421192', '0000336128', '0000626322', '0000925737', '0000990926', '0000715767', '0000509676', '0000306571', '0000108941', '0000354402', '0000724089', '0001015136', '0000065253', '0000150573', '0000996658', '0000245347','0000876881', '0000776724', '0001014213', '0000706877', '0000626470', '0000745235', '0000874742', '0000217989', '0000425994']
            # tr_keys = ['Tr_091', 'Tr_002', 'Tr_018', 'Tr_040', 'Tr_007', 'Tr_088', 'Tr_055', 'Tr_080', 'Tr_066', 'Tr_026',
            #            'Tr_067', 'Tr_017', 'Tr_077', 'Tr_057', 'Tr_090', 'Tr_064', 'Tr_051', 'Tr_042', 'Tr_073', 'Tr_029',
            #            'Tr_033', 'Tr_047', 'Tr_095', 'Tr_041', 'Tr_009', 'Tr_078', 'Tr_065', 'Tr_031', 'Tr_045', 'Tr_084',
            #            'Tr_036', 'Tr_019', 'Tr_049', 'Tr_021', 'Tr_037', 'Tr_022', 'Tr_060', 'Tr_068', 'Tr_052', 'Tr_086',
            #            'Tr_089', 'Tr_016', 'Tr_010', 'Tr_079', 'Tr_050', 'Tr_005', 'Tr_043', 'Tr_076', 'Tr_003', 'Tr_025',
            #            'Tr_008', 'Tr_048', 'Tr_082', 'Tr_032', 'Tr_069', 'Tr_054', 'Tr_061', 'Tr_070', 'Tr_094', 'Tr_039',
            #            'Tr_006', 'Tr_063', 'Tr_092', 'Tr_035', 'Tr_038', 'Tr_015', 'Tr_056', 'Tr_087', 'Tr_013', 'Tr_062',
            #            'Tr_044', 'Tr_058', 'Tr_059', 'Tr_020', 'Tr_074', 'Tr_075', 'Tr_024', 'Tr_004', 'Tr_028', 'Tr_099',
            #            'Tr_046', 'Tr_034', 'Tr_001', 'Tr_053', 'Tr_081', 'Tr_012', 'Tr_072', 'Tr_071', 'Tr_098', 'Tr_014',
            #            'Tr_030', 'Tr_096', 'Tr_097', 'Tr_083', 'Tr_093', 'Tr_085', 'Tr_023', 'Tr_027']
            # val_keys = ['Ts_071', 'Ts_020', 'Ts_090', 'Ts_024', 'Ts_056', 'Ts_099', 'Ts_026', 'Ts_097', 'Ts_021',
            #             'Ts_044', 'Ts_051', 'Ts_013', 'Ts_005', 'Ts_011', 'Ts_039', 'Ts_063', 'Ts_095', 'Ts_047',
            #             'Ts_004', 'Ts_089', 'Ts_082', 'Ts_076', 'Ts_098', 'Ts_066', 'Ts_079', 'Ts_041', 'Ts_088',
            #             'Ts_074', 'Ts_094', 'Ts_058', 'Ts_032', 'Ts_002', 'Ts_025', 'Ts_073', 'Ts_007', 'Ts_014',
            #             'Ts_029', 'Ts_087', 'Ts_081', 'Ts_045', 'Ts_096', 'Ts_050', 'Ts_080', 'Ts_069', 'Ts_086',
            #             'Ts_067', 'Ts_078', 'Ts_091', 'Ts_031', 'Ts_023', 'Ts_043', 'Ts_093', 'Ts_092', 'Ts_077',
            #             'Ts_070', 'Ts_057', 'Ts_003', 'Ts_062', 'Ts_040', 'Ts_064', 'Ts_016', 'Ts_046', 'Ts_061',
            #             'Ts_060', 'Ts_001', 'Ts_037', 'Ts_085', 'Ts_049', 'Ts_075', 'Ts_065', 'Ts_055', 'Ts_022',
            #             'Ts_048', 'Ts_042', 'Ts_019', 'Ts_035', 'Ts_072', 'Ts_010', 'Ts_006', 'Ts_052', 'Ts_033',
            #             'Ts_008', 'Ts_053', 'Ts_018', 'Ts_015', 'Ts_059', 'Ts_034', 'Ts_083', 'Ts_027', 'Ts_009',
            #             'Ts_084', 'Ts_028', 'Ts_054', 'Ts_068', 'Ts_012', 'Ts_038', 'Ts_017', 'Ts_036', 'Ts_030']
            # tr_keys = ['Tr_091', 'Tr_018', 'Tr_040', 'Tr_007', 'Tr_088', 'Tr_055', 'Tr_080', 'Tr_066', 'Tr_026',
            #             'Tr_067', 'Tr_017', 'Tr_077', 'Tr_057', 'Tr_090', 'Tr_064', 'Tr_051', 'Tr_042', 'Tr_073',
            #             'Tr_029', 'Tr_033', 'Tr_047', 'Tr_095', 'Tr_041', 'Tr_009', 'Tr_078', 'Tr_065', 'Tr_031',
            #             'Tr_045', 'Tr_084', 'Tr_036', 'Tr_019', 'Tr_049', 'Tr_021', 'Tr_037', 'Tr_022', 'Tr_060',
            #             'Tr_068', 'Tr_052', 'Tr_086', 'Tr_089', 'Tr_016', 'Tr_079', 'Tr_050', 'Tr_005', 'Tr_043',
            #             'Tr_076', 'Tr_003', 'Tr_025', 'Tr_008', 'Tr_048', 'Tr_082', 'Tr_032', 'Tr_069', 'Tr_054',
            #             'Tr_061', 'Tr_070', 'Tr_094', 'Tr_039', 'Tr_006', 'Tr_063', 'Tr_092', 'Tr_035', 'Tr_038',
            #             'Tr_015', 'Tr_056', 'Tr_087', 'Tr_013', 'Tr_062', 'Tr_044', 'Tr_058', 'Tr_059', 'Tr_020',
            #             'Tr_074', 'Tr_075', 'Tr_024', 'Tr_004', 'Tr_028', 'Tr_099', 'Tr_046', 'Tr_034', 'Tr_001',
            #             'Tr_053', 'Tr_081', 'Tr_012', 'Tr_072', 'Tr_071', 'Tr_098', 'Tr_014', 'Tr_030', 'Tr_096',
            #             'Tr_097', 'Tr_083', 'Tr_093', 'Tr_085', 'Tr_023', 'Tr_027']
            # tr_keys = ['Ts_071', 'Ts_020', 'Tr_091', 'Ts_090', 'Ts_100', 'Ts_024', 'Ts_149', 'Tr_088', 'Ts_056',
            #             'Ts_099', 'Ts_026', 'Tr_055', 'Ts_126', 'Ts_148', 'Ts_097', 'Ts_021', 'Tr_080', 'Ts_044',
            #             'Ts_113', 'Ts_136', 'Ts_051', 'Ts_130', 'Tr_066', 'Ts_145', 'Ts_111', 'Ts_013', 'Tr_067',
            #             'Ts_152', 'Ts_120', 'Tr_077', 'Ts_005', 'Ts_011', 'Ts_139', 'Ts_039', 'Ts_122', 'Tr_057',
            #             'Tr_090', 'Tr_064', 'Ts_063', 'Ts_095', 'Ts_047', 'Ts_004', 'Ts_089', 'Tr_051', 'Ts_082',
            #             'Tr_073', 'Ts_076', 'Ts_098', 'Ts_066', 'Ts_079', 'Ts_115', 'Ts_125', 'Ts_041', 'Ts_140',
            #             'Ts_132', 'Tr_095', 'Ts_088', 'Ts_074', 'Ts_094', 'Ts_058', 'Ts_032', 'Ts_002', 'Tr_078',
            #             'Tr_065', 'Tr_084', 'Ts_025', 'Ts_073', 'Ts_007', 'Ts_118', 'Ts_014', 'Ts_129', 'Ts_121',
            #             'Ts_103', 'Ts_029', 'Ts_087', 'Ts_147', 'Ts_137', 'Ts_110', 'Tr_060', 'Tr_068', 'Ts_123',
            #             'Tr_052', 'Ts_081', 'Ts_045', 'Tr_086', 'Tr_089', 'Ts_109', 'Ts_096', 'Ts_050', 'Tr_079',
            #             'Ts_080', 'Ts_114', 'Ts_102', 'Ts_069', 'Ts_119', 'Tr_076', 'Ts_086', 'Ts_067', 'Ts_078',
            #             'Ts_091', 'Ts_141', 'Ts_128', 'Ts_031', 'Tr_082', 'Ts_023', 'Ts_043', 'Ts_093', 'Ts_092',
            #             'Ts_077', 'Ts_144', 'Tr_069', 'Tr_054', 'Tr_061', 'Tr_070', 'Tr_094', 'Ts_070', 'Ts_057',
            #             'Ts_138', 'Ts_003', 'Ts_062', 'Ts_040', 'Tr_063', 'Tr_092', 'Ts_064', 'Ts_016', 'Ts_146',
            #             'Ts_046', 'Ts_061', 'Ts_060', 'Ts_001', 'Ts_037', 'Tr_056', 'Ts_105', 'Ts_131', 'Tr_087',
            #             'Ts_108', 'Ts_085', 'Tr_062', 'Ts_142', 'Ts_049', 'Ts_075', 'Ts_065', 'Tr_058', 'Tr_059',
            #             'Ts_107', 'Ts_055', 'Tr_074', 'Tr_075', 'Ts_134', 'Ts_022', 'Ts_112', 'Tr_099', 'Ts_048',
            #             'Ts_042', 'Ts_019', 'Ts_035', 'Ts_072', 'Ts_010', 'Ts_133', 'Ts_006', 'Ts_052', 'Ts_033',
            #             'Ts_104', 'Ts_008', 'Ts_053', 'Tr_053', 'Ts_116', 'Tr_081', 'Ts_018', 'Tr_072', 'Tr_071',
            #             'Tr_098', 'Ts_015', 'Ts_143', 'Ts_117', 'Ts_059', 'Ts_124', 'Ts_135', 'Ts_034', 'Ts_083',
            #             'Ts_106', 'Ts_101', 'Tr_096', 'Tr_097', 'Ts_027', 'Ts_009', 'Ts_084', 'Tr_083', 'Ts_028',
            #             'Tr_093', 'Ts_054', 'Ts_150', 'Ts_068', 'Tr_085', 'Ts_012', 'Ts_038', 'Ts_017', 'Ts_036',
            #             'Ts_030', 'Ts_127', 'Ts_151']
            # ============================ LIASD闁轰胶澧楀畵渚€姊?缂侇喖澧介崑褔姊奸懜娈垮斀 =============================
            # tr_keys = ['Ts_071', 'Tr_091', 'Ts_149', 'Tr_088', 'Ts_056', 'Ts_099', 'Tr_055', 'Ts_126', 'Ts_148',
            #             'Ts_097', 'Tr_080', 'Ts_113', 'Ts_136', 'Tr_066', 'Ts_145', 'Ts_111', 'Ts_013', 'Tr_067',
            #             'Ts_152', 'Tr_077', 'Ts_011', 'Ts_139', 'Ts_122', 'Tr_057', 'Tr_090', 'Tr_064', 'Ts_063',
            #             'Ts_095', 'Ts_089', 'Tr_051', 'Ts_082', 'Tr_073', 'Ts_076', 'Ts_098', 'Ts_066', 'Ts_079',
            #             'Ts_115', 'Ts_125', 'Ts_132', 'Tr_095', 'Ts_088', 'Ts_074', 'Ts_094', 'Ts_058', 'Tr_078',
            #             'Tr_065', 'Tr_084', 'Ts_073', 'Ts_007', 'Ts_118', 'Ts_014', 'Ts_129', 'Ts_121', 'Ts_103',
            #             'Ts_087', 'Ts_147', 'Ts_137', 'Tr_060', 'Tr_068', 'Ts_123', 'Tr_052', 'Tr_086', 'Tr_089',
            #             'Ts_109', 'Ts_096', 'Tr_079', 'Ts_114', 'Ts_102', 'Ts_069', 'Ts_119', 'Tr_076', 'Ts_086',
            #             'Ts_067', 'Ts_078', 'Ts_091', 'Ts_141', 'Ts_128', 'Tr_082', 'Ts_093', 'Ts_092', 'Ts_077',
            #             'Ts_144', 'Tr_069', 'Tr_054', 'Tr_061', 'Tr_070', 'Tr_094', 'Ts_057', 'Ts_138', 'Ts_062',
            #             'Tr_063', 'Tr_092', 'Ts_064', 'Ts_016', 'Ts_146', 'Ts_061', 'Tr_056', 'Ts_105', 'Ts_131',
            #             'Tr_087', 'Ts_108', 'Ts_085', 'Tr_062', 'Ts_142', 'Ts_075', 'Ts_065', 'Tr_058', 'Tr_059',
            #             'Ts_107', 'Ts_055', 'Tr_074', 'Tr_075', 'Ts_134', 'Ts_112', 'Tr_099', 'Ts_072', 'Ts_133',
            #             'Ts_006',  'Ts_104', 'Ts_008', 'Ts_053', 'Tr_053', 'Ts_116', 'Tr_081', 'Tr_072',
            #             'Tr_071', 'Tr_098', 'Ts_015', 'Ts_143', 'Ts_117', 'Ts_059', 'Ts_124', 'Ts_135', 'Ts_106',
            #             'Ts_101', 'Tr_096', 'Tr_097', 'Tr_083', 'Tr_093', 'Ts_054', 'Ts_068', 'Tr_085', 'Ts_012',
            #             'Ts_127', 'Ts_151']
            # ============================ LIASD闁轰胶澧楀畵渚€姊?缂侇喖澧介崑褔姊奸懜娈垮斀 =============================
            # tr_keys = ['Ts_071', 'Ts_020', 'Ts_044', 'Ts_047', 'Ts_089', 'Ts_076', 'Ts_066', 'Ts_079', 'Ts_041',
            #            'Ts_088', 'Ts_007', 'Ts_014', 'Ts_087', 'Ts_045', 'Ts_050', 'Ts_091', 'Ts_077', 'Ts_040',
            #            'Ts_016', 'Ts_046', 'Ts_085', 'Ts_055', 'Ts_048', 'Ts_072', 'Ts_010', 'Ts_033', 'Ts_053',
            #            'Ts_059', 'Ts_083', 'Ts_054', 'Ts_068']
            # ============================ LIASD闁轰胶澧楀畵渚€姊?缂侇喖澧介崑褔姊奸懜娈垮斀 =============================
            # val_keys = ['Ts_071', 'Tr_091', 'Ts_149', 'Tr_088', 'Ts_056', 'Ts_099', 'Tr_055', 'Ts_126', 'Ts_148',
            #             'Ts_097', 'Tr_080', 'Ts_113', 'Ts_136', 'Tr_066', 'Ts_145', 'Ts_111', 'Ts_013', 'Tr_067',
            #             'Ts_152', 'Tr_077', 'Ts_011', 'Ts_139', 'Ts_122', 'Tr_057', 'Tr_090', 'Tr_064', 'Ts_063',
            #             'Ts_095', 'Ts_089', 'Tr_051', 'Ts_082', 'Tr_073', 'Ts_076', 'Ts_098', 'Ts_066', 'Ts_079',
            #             'Ts_115', 'Ts_125', 'Ts_132', 'Tr_095', 'Ts_088', 'Ts_074', 'Ts_094', 'Ts_058', 'Tr_078',
            #             'Tr_065', 'Tr_084', 'Ts_073', 'Ts_007', 'Ts_118', 'Ts_014', 'Ts_129', 'Ts_121', 'Ts_103',
            #             'Ts_087', 'Ts_147', 'Ts_137', 'Tr_060', 'Tr_068', 'Ts_123', 'Tr_052', 'Tr_086', 'Tr_089',
            #             'Ts_109', 'Ts_096', 'Tr_079', 'Ts_114', 'Ts_102', 'Ts_069', 'Ts_119', 'Tr_076', 'Ts_086',
            #             'Ts_067', 'Ts_078', 'Ts_091', 'Ts_141', 'Ts_128', 'Tr_082', 'Ts_093', 'Ts_092', 'Ts_077',
            #             'Ts_144', 'Tr_069', 'Tr_054', 'Tr_061', 'Tr_070', 'Tr_094', 'Ts_057', 'Ts_138', 'Ts_062',
            #             'Tr_063', 'Tr_092', 'Ts_064', 'Ts_016', 'Ts_146', 'Ts_061', 'Tr_056', 'Ts_105', 'Ts_131',
            #             'Tr_087', 'Ts_108', 'Ts_085', 'Tr_062', 'Ts_142', 'Ts_075', 'Ts_065', 'Tr_058', 'Tr_059',
            #             'Ts_107', 'Ts_055', 'Tr_074', 'Tr_075', 'Ts_134', 'Ts_112', 'Tr_099', 'Ts_072', 'Ts_133',
            #             'Ts_006',  'Ts_104', 'Ts_008', 'Ts_053', 'Tr_053', 'Ts_116', 'Tr_081', 'Tr_072',
            #             'Tr_071', 'Tr_098', 'Ts_015', 'Ts_143', 'Ts_117', 'Ts_059', 'Ts_124', 'Ts_135', 'Ts_106',
            #             'Ts_101', 'Tr_096', 'Tr_097', 'Tr_083', 'Tr_093', 'Ts_054', 'Ts_068', 'Tr_085', 'Ts_012',
            #             'Ts_127', 'Ts_151']

        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append({})
                    splits[-1]['train'] = list(train_keys)
                    splits[-1]['val'] = list(test_keys)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetDataset(self.preprocessed_dataset_folder, tr_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0)
        dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0)
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data=3, order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size, dim)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            print('閻犙囶暒缁ㄢ剝娼诲▎鎾虫')
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr, transform=tr_transforms,
                                             num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
                                           transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                                           num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
                                           wait_time=0.02)

        print(len(mt_gen_train))
        print(len(mt_gen_val))
        return mt_gen_train, mt_gen_val

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        print('=============')
        print(len(dataset_tr))
        print(len(dataset_val))
        print(self.oversample_foreground_percent)
        print('=============')

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        else:
            dl_tr = nnUNetDataLoader3D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader3D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        return dl_tr, dl_val

    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        tr_transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=None,
            do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
            do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
            p_rot_per_axis=1,  # todo experiment with this
            do_scale=True, scale=(0.7, 1.4),
            border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
            random_crop=False,  # random cropping is part of our dataloaders
            p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False  # todo experiment with this
        ))

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=ignore_axes))
        tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                               mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            tr_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(foreground_labels), 0)),
                p_per_sample=0.4,
                key="data",
                strel_size=(1, 8),
                p_per_label=1))
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15))

        tr_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            tr_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                       if ignore_label is not None else regions,
                                                                       'target', 'target'))

        if deep_supervision_scales is not None:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                              output_key='target'))
        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms(deep_supervision_scales: Union[List, Tuple],
                                  is_cascaded: bool = False,
                                  foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                  regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                  ignore_label: int = None) -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            val_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))

        val_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            val_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                        if ignore_label is not None else regions,
                                                                        'target', 'target'))

        if deep_supervision_scales is not None:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))

        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            self.network.module.decoder.deep_supervision = enabled
        else:
            self.network.decoder.deep_supervision = enabled

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(True)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
            self.print_to_log_file('unpacking done...')

        if self.is_ddp:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()


    def on_train_end(self):
        # dirty hack because on_epoch_end increments the epoch counter and this is executed afterwards.
        # This will lead to the wrong current epoch to be stored
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        # now we can delete latest
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None:
                self.dataloader_train._finish()
            if self.dataloader_val is not None:
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)

    def on_validation_epoch_start(self):
        self.network.eval()

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution
        output = output[0]
        target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # todo find a solution for this stupid shit
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            # 闁哄瀚紓鎾寸┍濠靛棛鎽犻悹渚灠缁?            checkpoint_path = join(self.output_folder, 'checkpoint_best.pth')

            # 闁瑰灚鎸稿畵鍐╃┍濠靛棛鎽犻悹渚灠缁?            self.print_to_log_file(f"Saving checkpoint to: {checkpoint_path}")

            # 濞ｅ洦绻傞悺銊╁级閸愵喖娅?            self.save_checkpoint(checkpoint_path)

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_gpu=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for k in dataset_val.keys():
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                output_filename_truncated = join(validation_output_folder, k)

                try:
                    prediction = predictor.predict_sliding_window_return_logits(data)
                except RuntimeError:
                    predictor.perform_everything_on_gpu = False
                    prediction = predictor.predict_sliding_window_return_logits(data)
                    predictor.perform_everything_on_gpu = True

                prediction = prediction.cpu()

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )
                # for debug purposes
                # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                #              output_filename_truncated, save_probabilities)

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = nnUNetDataset(expected_preprocessed_folder, [k],
                                                num_images_properties_loading_threshold=0)
                            d, s, p = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file = join(output_folder, k + '.npz')

                        # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                        #                   self.dataset_json)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json),
                            )
                        ))

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

    def perform_actual_validation_every_epoch(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_gpu=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for k in dataset_val.keys():
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                output_filename_truncated = join(validation_output_folder, k)

                try:
                    prediction = predictor.predict_sliding_window_return_logits(data)
                except RuntimeError:
                    predictor.perform_everything_on_gpu = False
                    prediction = predictor.predict_sliding_window_return_logits(data)
                    predictor.perform_everything_on_gpu = True

                prediction = prediction.cpu()

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )
                # for debug purposes
                # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                #              output_filename_truncated, save_probabilities)

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = nnUNetDataset(expected_preprocessed_folder, [k],
                                                num_images_properties_loading_threshold=0)
                            d, s, p = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file = join(output_folder, k + '.npz')

                        # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                        #                   self.dataset_json)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json),
                            )
                        ))

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

        return metrics['foreground_mean']["Dice"]

    def run_training(self):
        self.on_train_start()


        for name, param in self.network.named_parameters():
            if 'decoder.seg_layers' in name:
                param.requires_grad = True
                print(name)
            elif 'decoder.stages.4.convs.1' in name:
                param.requires_grad = True
            elif 'finetune' in name:
                param.requires_grad = True
                print(name)
            else:
                param.requires_grad = False

        print('Loaded pretrained model and froze selected layers for finetune')

        best_dice = -1

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

            if epoch % 5 == 0:
                pred_dice = self.perform_actual_validation_every_epoch(False)
                if pred_dice > best_dice:
                    best_dice = pred_dice
                    # self.save_checkpoint(join(self.output_folder, 'checkpoint_zyh_best_' + str(epoch) + '.pth'))
                    self.save_checkpoint(join(self.output_folder, 'checkpoint_zyh_best.pth'))
                    self.print_to_log_file(f"pred'best dice: {np.round(best_dice, decimals=4)}")

        self.on_train_end()
