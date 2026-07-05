import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import time
from pyiqa import create_metric
from models.archs import define_network
from models.base_model import BaseModel
from utils import get_root_logger, imwrite, tensor2img


class TestModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))

        self.metrics = {}
        val_opt = opt.get('val', {})
        if val_opt.get('metrics') is not None:
            for name, metric_opt in val_opt['metrics'].items():
                mopt = deepcopy(metric_opt)
                metric_type = mopt.pop('type').lower()
                mopt['device'] = self.device

                if metric_type in ('psnr', 'ssim'):
                    mopt.setdefault('test_y_channel', True)
                    mopt.setdefault('color_space', 'ycbcr')

                self.metrics[name] = create_metric(metric_type, **mopt)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        self.ref = data['ref'].to(self.device)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output, self.output_low = self.net_g(self.lq, self.ref)

    def _sync_device(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def test_speed(self, times_per_img=50, size=None):
        if size is not None:
            self.ref = torch.empty(
                (1, self.ref.shape[1], size[0], size[1]),
                dtype=self.ref.dtype,
                device=self.lq.device)

        sum_time1, sum_time2 = 0, 0
        self.net_g.eval()
        with torch.no_grad():
            self._sync_device()
            start = time.time()
            for _ in range(times_per_img):
                self.output, self.output_low = self.net_g(self.lq, self.ref)
            self._sync_device()
            self.duration = (time.time() - start)
            self.avg_time1 = sum_time1
            self.avg_time2 = sum_time2

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
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
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
                                             f'{img_name}_{current_iter}.jpg')
                else:
                    suffix = self.opt['val'].get('suffix')
                    if suffix:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{suffix}.jpg')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.jpg')
                for item in visual_imgs:
                    imwrite(visual_imgs[item],
                            save_img_path.replace('.jpg', f'_{item}.jpg'))

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def nondist_validation_speed(self, dataloader, times_per_img, num_imgs, size):
        logger = get_root_logger()
        total_duration = 0
        total_time1 = 0
        total_time2 = 0
        count = 0
        for idx, val_data in enumerate(dataloader):
            if count >= num_imgs:
                break
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test_speed(times_per_img, size=size)
            total_duration += self.duration
            total_time1 += self.avg_time1
            total_time2 += self.avg_time2
            count += 1
            logger.info(
                f'{idx} Testing {img_name} '
                f'(shape: {self.lq.shape[2]} * {self.lq.shape[3]}) '
                f'duration: {self.duration}')

        if count == 0:
            logger.warning('No images were tested for speed.')
            return

        logger.info(f'average duration is {total_duration / count} seconds')
        logger.info(f'average time1 is {total_time1 / count} seconds')
        logger.info(f'average time2 is {total_time2 / count} seconds')

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
