import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from pyiqa import create_metric
from models.archs import define_network
from models.base_model import BaseModel
from utils import get_root_logger, imwrite, tensor2img


loss_module = importlib.import_module('models.losses')


class TrainModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.net_d = None
        self.use_gan = False

        # Define networks.
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        if self.opt.get('network_d'):
            self.net_d = define_network(deepcopy(self.opt['network_d']))
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)

        # Load pretrained models.
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', False))
        if self.opt.get('network_d'):
            load_path = self.opt['path'].get('pretrain_network_d', None)
            if load_path is not None:
                self.load_network(self.net_d, load_path,
                                  self.opt['path'].get('strict_load_d', False))

        if self.is_train:
            self.init_training_settings()

        self.metrics = {}
        val_opt = opt['val']
        if val_opt.get('metrics') is not None:
            for name, metric_opt in val_opt['metrics'].items():
                mopt = deepcopy(metric_opt)
                metric_type = mopt.pop('type').lower()
                mopt['device'] = self.device

                if metric_type in ('psnr', 'ssim'):
                    mopt.setdefault('test_y_channel', True)
                    mopt.setdefault('color_space', 'ycbcr')

                self.metrics[name] = create_metric(metric_type, **mopt)

    def _build_loss(self, train_opt, opt_key, attr_name):
        if train_opt.get(opt_key):
            loss_type = train_opt[opt_key].pop('type')
            loss_cls = getattr(loss_module, loss_type)
            setattr(self, attr_name,
                    loss_cls(**train_opt[opt_key]).to(self.device))

    def init_training_settings(self):
        self.net_g.train()
        if self.net_d is not None:
            self.net_d.train()
        train_opt = self.opt['train']

        # define losses
        self._build_loss(train_opt, 'pixel_opt', 'cri_pix')
        self._build_loss(train_opt, 'pixel_color_opt', 'cri_pix_color')
        self._build_loss(train_opt, 'perceptual_opt', 'cri_perceptual')
        self._build_loss(train_opt, 'ssim_opt', 'cri_ssim')
        self._build_loss(train_opt, 'color_opt', 'cri_color')
        self._build_loss(train_opt, 'lpips_opt', 'cri_lpips')
        self._build_loss(train_opt, 'flow_opt', 'cri_flow')
        self._build_loss(train_opt, 'CoBi_opt', 'cri_CoBi')
        if self.net_d is not None and train_opt.get('gan_opt') is None:
            raise ValueError('network_d requires train.gan_opt.')
        if self.net_d is None and train_opt.get('gan_opt') is not None:
            raise ValueError('train.gan_opt requires network_d.')

        if self.net_d is not None:
            gan_opt = train_opt['gan_opt']
            gan_type = train_opt['gan_opt'].pop('type')
            cri_gan_cls = getattr(loss_module, gan_type)
            self.cri_gan = cri_gan_cls(
                gan_opt['gan_type'],
                loss_weight=gan_opt['gan_weight'],
                real_label_val=1.0,
                fake_label_val=0.0
            ).to(self.device)
            self.use_gan = True

        if train_opt.get('gp_opt'):
            self.gp_weight = train_opt['gp_opt'].pop('loss_weight')

        if self.use_gan:
            self.net_g_pretrain_steps = train_opt.get('net_g_pretrain_steps', 0)
            self.net_d_steps = train_opt.get('net_d_steps', 1) or 1
            self.net_d_init_steps = train_opt.get('net_d_init_steps', 0) or 0

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        logger = get_root_logger()
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                logger.info(f'add {k} to update list')
                optim_params.append(v)
            else:
                logger.warning(f'Params {k} will not be optimized.')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        elif optim_type == 'Muon':
            try:
                from muon import MuonWithAuxAdam
            except ImportError as exc:
                raise ImportError(
                    'Muon optimizer requires the standalone muon.py module. '
                    'Install/vendor that module or change train.optim_g.type '
                    'to Adam.'
                ) from exc
            lr = train_opt['optim_g']['lr']
            hidden_weights = [p for p in optim_params if p.ndim >= 2]
            hidden_gains_biases = [p for p in optim_params if p.ndim < 2]
            param_groups = [
                dict(params=hidden_weights, use_muon=True,
                     lr=10 * lr, weight_decay=0.001),
                dict(params=hidden_gains_biases, use_muon=False,
                     lr=lr, betas=(0.9, 0.99), weight_decay=0.001),
            ]
            self.optimizer_g = MuonWithAuxAdam(param_groups)
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supported yet.')
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        if self.net_d is not None:
            if train_opt.get('optim_d') is None:
                raise ValueError('network_d requires train.optim_d.')
            optim_type = train_opt['optim_d'].pop('type')
            if optim_type == 'Adam':
                self.optimizer_d = torch.optim.Adam(self.net_d.parameters(),
                                                    **train_opt['optim_d'])
            else:
                raise NotImplementedError(
                    f'optimizer {optim_type} is not supported yet.')
            self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        self.ref = data['ref'].to(self.device)
        self.gt_low = data['gt_low'].to(self.device)

    def _accumulate_generator_losses(self, loss_dict):
        l_g_total = 0
        if hasattr(self, 'cri_pix'):
            l_g_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix
        if hasattr(self, 'cri_pix_color'):
            l_g_pix_color = self.cri_pix_color(self.output_low, self.gt_low)
            l_g_total += l_g_pix_color
            loss_dict['l_g_pix_color'] = l_g_pix_color
        if hasattr(self, 'cri_perceptual'):
            l_g_perceptual = self.cri_perceptual(self.output_low,
                                                 self.gt_low)
            l_g_total += l_g_perceptual
            loss_dict['l_g_perceptual'] = l_g_perceptual
        if hasattr(self, 'cri_ssim'):
            l_g_ssim = self.cri_ssim(self.output_low, self.gt_low)
            l_g_total += l_g_ssim
            loss_dict['l_g_ssim'] = l_g_ssim
        if hasattr(self, 'cri_color'):
            l_g_color = self.cri_color(self.output, self.gt)
            l_g_total += l_g_color
            loss_dict['l_g_color'] = l_g_color
        if hasattr(self, 'cri_lpips'):
            l_g_lpips = self.cri_lpips(self.output, self.gt)
            l_g_total += l_g_lpips
            loss_dict['l_g_lpips'] = l_g_lpips
        if hasattr(self, 'cri_CoBi'):
            l_g_CoBi = self.cri_CoBi(self.output, self.gt)
            l_g_total += l_g_CoBi
            loss_dict['l_g_CoBi'] = l_g_CoBi
        if hasattr(self, 'cri_flow'):
            l_g_flow = self.cri_flow(self.aligned_lrs, self.mid_frame)
            l_g_total += l_g_flow
            loss_dict['l_g_flow'] = l_g_flow
        return l_g_total

    def _optimize_generator(self, loss_dict, use_gan_loss=False):
        self.optimizer_g.zero_grad()
        l_g_total = self._accumulate_generator_losses(loss_dict)

        if use_gan_loss:
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        l_g_total.backward()
        self.optimizer_g.step()

    def _optimize_discriminator(self, loss_dict):
        self.optimizer_d.zero_grad()
        for p in self.net_d.parameters():
            p.requires_grad = True

        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())

        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        l_d_total = l_d_real + l_d_fake
        l_d_total.backward()
        self.optimizer_d.step()

    def optimize_parameters(self, step):
        self.output, self.output_low = self.net_g(self.lq, self.ref)

        loss_dict = OrderedDict()
        if not self.use_gan:
            self._optimize_generator(loss_dict)
        elif step <= self.net_g_pretrain_steps:
            self._optimize_generator(loss_dict)
        else:
            self._optimize_discriminator(loss_dict)

            for p in self.net_d.parameters():
                p.requires_grad = False

            if (step - self.net_g_pretrain_steps) % self.net_d_steps == 0 and (
                    step - self.net_g_pretrain_steps) > self.net_d_init_steps:
                self._optimize_generator(loss_dict, use_gan_loss=True)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output, self.output_low = self.net_g(self.lq, self.ref)

        self.net_g.train()

    def _to_metric_tensor(self, image):
        return (torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
                .to(self.device).float() / 255.0)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        logger = get_root_logger()
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            visual_imgs = {}
            for item in visuals:
                visual_imgs[item] = tensor2img(visuals[item])
                # Release per-image tensors early to reduce validation memory.
                if hasattr(self, item):
                    delattr(self, item)
            torch.cuda.empty_cache()

            if with_metrics:
                for name, metric in self.metrics.items():
                    output_tensor = self._to_metric_tensor(visual_imgs['output'])
                    gt_tensor = self._to_metric_tensor(visual_imgs['gt'])
                    if self.opt['val'].get('crop_border'):
                        border_pixels = self.opt['val']['crop_border']
                        _, _, h, w = output_tensor.shape
                        output_tensor = output_tensor[:, :,
                                                      border_pixels:h - border_pixels,
                                                      border_pixels:w - border_pixels]
                        gt_tensor = gt_tensor[:, :,
                                              border_pixels:h - border_pixels,
                                              border_pixels:w - border_pixels]
                    result = metric(output_tensor, gt_tensor)
                    logger.info(f'{name}_{img_name}: {result.item()}')
                    self.metric_results[name] += result.item()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             str(current_iter),
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                for item in visual_imgs:
                    imwrite(visual_imgs[item],
                            save_img_path.replace('.png', f'_{item}.png'))

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        for item in self.opt['val']['visuals']:
            out_dict[item] = getattr(self, item).detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        if self.net_d is not None:
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
