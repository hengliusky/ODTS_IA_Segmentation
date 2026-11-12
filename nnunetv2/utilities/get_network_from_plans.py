import os

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn

# ============================ 两阶段网络接入 ============================
# Focus 阶段：普通 nnU-Net U-Net（无 DTS）
# Refinement 阶段：U-Net + DTS-H / DTS-V
from my_model.coarse_net import PlainConvUNet as FocusPlainConvUNet
from my_model.refinement_net import PlainConvUNet as RefinementPlainConvUNet

_FOCUS_ALIASES = {'focus', 'coarse', 'stage1', 'stage_1', '1'}
_REFINE_ALIASES = {'refinement', 'refine', 'dts', 'stage2', 'stage_2', '2'}

# Refinement 网络内部硬编码了 6 级 encoder 和 128^3 输入下的特征尺寸；
# 这些值必须与 plans 一致，否则 DTS 模块的通道/空间尺寸会对不上。
_REFINEMENT_EXPECTED_FEATURES = [32, 64, 128, 256, 320, 320]
_REFINEMENT_EXPECTED_PATCH = (128, 128, 128)


def _resolve_plainconvunet_variant(plans_manager: PlansManager):
    """决定 PlainConvUNet 到底用 Focus 还是 Refinement 版本。

    优先级：
    1. 环境变量 MY_NNUNET_NETWORK=focus|refinement（最明确）；
    2. dataset 名里包含 'refine'（兼容 Dataset508_*_refine_* 这类命名）；
    3. 默认 Focus，保持原有行为。
    """
    requested = os.environ.get('MY_NNUNET_NETWORK', '').strip().lower()
    if requested in _REFINE_ALIASES:
        return RefinementPlainConvUNet, 'refinement'
    if requested in _FOCUS_ALIASES:
        return FocusPlainConvUNet, 'focus'

    dataset_name = str(getattr(plans_manager, 'dataset_name', '') or '').lower()
    if 'refine' in dataset_name:
        return RefinementPlainConvUNet, 'refinement'
    return FocusPlainConvUNet, 'focus'


def get_network_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than infer it again in the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = configuration_manager.UNet_class_name

    # -------------------- 选择网络类 --------------------
    if segmentation_network_class_name in {'PlainConvUNetRefinement', 'PlainConvUNetDTS'}:
        network_class, variant_name = RefinementPlainConvUNet, 'refinement'
    elif segmentation_network_class_name == 'PlainConvUNetFocus':
        network_class, variant_name = FocusPlainConvUNet, 'focus'
    elif segmentation_network_class_name == 'ResidualEncoderUNet':
        network_class, variant_name = ResidualEncoderUNet, 'residual'
    elif segmentation_network_class_name == 'PlainConvUNet':
        network_class, variant_name = _resolve_plainconvunet_variant(plans_manager)
    else:
        raise AssertionError(
            'The network architecture specified by the plans file is non-standard (maybe your own?). '
            f'Got UNet_class_name={segmentation_network_class_name!r}. Supported values: '
            "'PlainConvUNet', 'PlainConvUNetFocus', 'PlainConvUNetRefinement', 'ResidualEncoderUNet'."
        )

    print(f'[my_nnunet] network variant: {variant_name} '
          f'(dataset={plans_manager.dataset_name}, UNet_class_name={segmentation_network_class_name})')

    # -------------------- 构建参数 --------------------
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    features_per_stage = [min(configuration_manager.UNet_base_num_features * 2 ** i,
                              configuration_manager.unet_max_num_features) for i in range(num_stages)]

    # Refinement 网络里 DTS 的通道数/特征图尺寸是写死的，不匹配时给出明确错误，
    # 而不是等到 forward 阶段出现难以定位的 shape mismatch。
    if network_class is RefinementPlainConvUNet:
        if num_stages != len(_REFINEMENT_EXPECTED_FEATURES):
            raise RuntimeError(
                f'my_model.refinement_net expects {len(_REFINEMENT_EXPECTED_FEATURES)} encoder stages, '
                f'but the plans have {num_stages}. Please train/infer the refinement stage with a 6-stage plans, '
                f'or update my_model/refinement_net.py accordingly.'
            )
        if tuple(configuration_manager.patch_size) != _REFINEMENT_EXPECTED_PATCH:
            raise RuntimeError(
                f'my_model.refinement_net expects a {_REFINEMENT_EXPECTED_PATCH} patch (DTS sizes are hardcoded), '
                f'but the plans specify {tuple(configuration_manager.patch_size)}.'
            )
        if list(features_per_stage) != _REFINEMENT_EXPECTED_FEATURES:
            raise RuntimeError(
                f'my_model.refinement_net expects features_per_stage={_REFINEMENT_EXPECTED_FEATURES}, '
                f'but the plans produce {list(features_per_stage)}.'
            )

    conv_or_blocks_per_stage = {
        'n_conv_per_stage'
        if network_class != ResidualEncoderUNet else 'n_blocks_per_stage':
            configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }
    kwargs_key = 'ResidualEncoderUNet' if network_class is ResidualEncoderUNet else 'PlainConvUNet'

    # -------------------- 构建网络 --------------------
    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[kwargs_key]
    )

    model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)
    return model
