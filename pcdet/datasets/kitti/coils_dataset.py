import glob
import numpy as np
from pathlib import Path
from ..dataset import DatasetTemplate

class CoilsDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        '''
        Args:
        root_path:
        dataset_cfg:
        class_names:
        training:
        logger:

        '''
        super().__init__(
        dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        points_file_list = glob.glob(str(self.root_path / 'train/points' / '*.txt'))
        labels_file_list = glob.glob(str(self.root_path / 'train/labels' / '*.txt'))
        points_file_list.sort()
        labels_file_list.sort()
        self.sample_file_list = points_file_list
        self.samplelabel_file_list = labels_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        sample_idx = Path(self.sample_file_list[index]).stem  # 0000.txt -> 0000 样本id(文件编号) n：点的个数 m：标注的个数
        points = np.loadtxt(self.sample_file_list[index], dtype=np.float32).reshape(-1, 3)  # 每个点云文件里的所有点 n*3
        points_label = np.loadtxt(self.samplelabel_file_list[index], dtype=np.float32).reshape(-1, 7) # 每个点云标注文件里的所有点 m*7
        gt_names = np.array(['Coil']*points_label.shape[0])

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'gt_names': gt_names,
            'gt_boxes': points_label
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict