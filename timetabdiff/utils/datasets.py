import numpy as np
import os, pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
from sdv.metadata import MultiTableMetadata
from rdt import HyperTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from rdt.transformers.categorical import LabelEncoder

def save_dict(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
class TimeTabDiffDataset(Dataset):
    def __init__(self, data_dir, tablename, is_train=True, seq_len=100, fk=None):
        self.data_dir = data_dir
        self.tablename = tablename
        self.is_train = is_train
        self.seq_len = seq_len
        self.info_path = os.path.join(data_dir, f'{tablename}_info.pkl')
        self.info = None

        meta_path = os.path.join(data_dir, 'metadata.json')
        self.multi_metadata = MultiTableMetadata.load_from_json(meta_path)
        self.table_metadata = self.multi_metadata.tables[tablename]
        self.pk = self.table_metadata.primary_key
        
        if fk:
            self.fk = fk
        else:
            for rel in self.multi_metadata.relationships:
                if rel['child_table_name'] == tablename:
                    self.fk = rel['child_foreign_key']
                    self.parent_tablename = rel['parent_table_name']
                    break
        
        if not hasattr(self, 'parent_tablename'):
            raise ValueError(f"Parent table not found for {tablename}. Check metadata.")

        self.columns_info = self.table_metadata.columns
        self.num_cols, self.cat_cols, self.dt_cols = [], [], []

        for col, info in self.columns_info.items():
            if col in [self.pk, self.fk]: 
                continue
            sdtype = info['sdtype']
            if sdtype == 'numerical': self.num_cols.append(col)
            elif sdtype == 'datetime': self.dt_cols.append(col)
            elif sdtype in ['categorical', 'boolean', 'id']: self.cat_cols.append(col)
        self.reordered_cols = self.dt_cols + self.num_cols + self.cat_cols
        
        # 데이터 로드
        self.raw_df = pd.read_csv(os.path.join(data_dir, f'{tablename}.csv'))
        self.parent_df = pd.read_csv(os.path.join(data_dir, f'{self.parent_tablename}.csv'))

        # datetime -> 일 단위 정수 변환
        for col in self.dt_cols:
            self.raw_df[col] = pd.to_datetime(self.raw_df[col]).dt.day.astype('int64')
        for col in self.cat_cols:
            self.raw_df[col] = self.raw_df[col].astype(str)

        parent_meta = self.multi_metadata.tables[self.parent_tablename]
        self.parent_pk = parent_meta.primary_key
        self.parent_num = [c for c, i in parent_meta.columns.items() if i['sdtype'] in ['numerical', 'datetime'] and c != self.parent_pk]
        self.parent_cat = [c for c, i in parent_meta.columns.items() if i['sdtype'] in ['categorical', 'boolean'] and c != self.parent_pk]
        for col in self.parent_num:
            if parent_meta.columns[col]['sdtype'] == 'datetime':
                self.parent_df[col] = pd.to_datetime(self.parent_df[col]).dt.day.astype('int64')

        self.ht = None
        self.parent_ht = None
        self.parent_map = {}
        self.data_tensor = None
        self.indices = None
        self.id_series = None
        self.d_parent = 0
        self.num_classes = []
        
        self._preprocess_and_group()
        
        
    def _preprocess_and_group(self):
        # Child num
        child_num_cols = self.dt_cols + self.num_cols
        # self.scaler_child = QuantileTransformer(
        #     output_distribution="uniform", n_quantiles=1000, subsample=int(1e9)
        # )
        self.scaler_child = MinMaxScaler(
            feature_range=(-1, 1)
        )
        if self.is_train:
            child_num_tf = pd.DataFrame(
                self.scaler_child.fit_transform(self.raw_df[child_num_cols]),
                columns=child_num_cols,
            )
        else:
            self.scaler_child = load_dict(os.path.join(self.data_dir, f"{self.tablename}_scaler.pkl"))
            child_num_tf = pd.DataFrame(
                self.scaler_child.transform(self.raw_df[child_num_cols]),
                columns=child_num_cols,
            )

        # Child cat (LabelEncoder)
        self.ht = HyperTransformer()
        self.ht.detect_initial_config(self.raw_df[self.cat_cols])
        for col in self.cat_cols:
            self.ht.update_transformers({col: LabelEncoder()})
        if self.is_train:
            child_cat_tf = self.ht.fit_transform(self.raw_df[self.cat_cols])
        else:
            self.ht = load_dict(os.path.join(self.data_dir, f"{self.tablename}_ht.pkl"))
            child_cat_tf = self.ht.transform(self.raw_df[self.cat_cols])

        child_tf = pd.concat([child_num_tf, child_cat_tf], axis=1)[self.reordered_cols]
        if self.is_train:
            save_dict(self.ht, os.path.join(self.data_dir, f"{self.tablename}_ht.pkl"))
            save_dict(self.scaler_child, os.path.join(self.data_dir, f"{self.tablename}_qt.pkl"))

        # Parent
        parent_num_cols = self.parent_num
        # self.scaler_parent = QuantileTransformer(
        #     output_distribution="uniform", n_quantiles=1000, subsample=int(1e9)
        # )
        self.scaler_parent = MinMaxScaler(
            feature_range=(-1, 1)
        )
        if self.is_train:
            parent_num_tf = pd.DataFrame(
                self.scaler_parent.fit_transform(self.parent_df[parent_num_cols]),
                columns=parent_num_cols,
            )
        else:
            self.scaler_parent = load_dict(os.path.join(self.data_dir, f"{self.parent_tablename}_scaler.pkl"))
            parent_num_tf = pd.DataFrame(
                self.scaler_parent.transform(self.parent_df[parent_num_cols]),
                columns=parent_num_cols,
            )
            
        parent_cols_cat = self.parent_cat
        self.parent_ht = HyperTransformer()
        self.parent_ht.detect_initial_config(self.parent_df[parent_cols_cat])
        for col in parent_cols_cat:
            self.parent_ht.update_transformers({col: LabelEncoder()})
        if self.is_train:
            parent_cat_tf = self.parent_ht.fit_transform(self.parent_df[parent_cols_cat])
        else:
            self.parent_ht = load_dict(os.path.join(self.data_dir, f"{self.parent_tablename}_ht.pkl"))
            parent_cat_tf = self.parent_ht.transform(self.parent_df[parent_cols_cat])

        parent_tf = pd.concat([parent_num_tf, parent_cat_tf], axis=1)[self.parent_num + self.parent_cat]
        if self.is_train:
            save_dict(self.parent_ht, os.path.join(self.data_dir, f"{self.parent_tablename}_ht.pkl"))
            save_dict(self.scaler_parent, os.path.join(self.data_dir, f"{self.parent_tablename}_scaler.pkl"))

        parent_vecs = parent_tf[self.parent_num + self.parent_cat].to_numpy(dtype=np.float32)
        parent_ids = self.parent_df[self.parent_pk].values
        self.parent_map = {pid: torch.tensor(vec) for pid, vec in zip(parent_ids, parent_vecs)}
        self.d_parent = len(parent_vecs[0])

        # sequence grouping
        sort_cols = [self.fk] + ([self.dt_cols[0]] if self.dt_cols else [])
        child_tf[self.fk] = self.raw_df[self.fk].values
        df_sorted = child_tf.sort_values(sort_cols)
        self.id_series = df_sorted[self.fk].values

        data_np = df_sorted[self.reordered_cols].to_numpy(dtype=np.float32)
        self.data_tensor = torch.tensor(data_np)

        indices = []
        _, start_idx = np.unique(self.id_series, return_index=True)
        start_idx = np.sort(start_idx)
        for i, s in enumerate(start_idx):
            e = start_idx[i + 1] if i < len(start_idx) - 1 else len(data_np)
            indices.append((s, e))
        self.indices = indices

        self.d_numerical = len(self.dt_cols) + len(self.num_cols)
        self.num_classes = [int(df_sorted[c].max()) + 1 for c in self.cat_cols]
        
    def get_info(self):
        return {
            "num_numerical_features": self.d_numerical,
            "num_classes": self.num_classes,
            "num_features": len(self.reordered_cols),
            "parent_dim": self.d_parent
        }
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start, end = self.indices[idx]
        seq_data = self.data_tensor[start:end]
        # 현재 시퀀스의 FK 찾아서 부모 벡터 가져오기
        current_fk = self.id_series[start]
        parent_vec = self.parent_map.get(current_fk, torch.zeros(self.d_parent))
        
        if seq_data.shape[0] > self.seq_len:
            seq_data = seq_data[-self.seq_len:]
        
        actual_len = seq_data.shape[0]
        final_data = torch.zeros((self.seq_len, seq_data.shape[1]), dtype=torch.float32)
        final_data[:actual_len, :] = seq_data
        
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        mask[:actual_len] = True

        return {
            "data": final_data,
            "mask": mask,
            "cond": parent_vec
        }
        
    def inverse_transform_child(self, df):
        if isinstance(df, torch.Tensor):
            df = df.detach().cpu().numpy()

        B, S, F = df.shape
        X_flat = df.reshape(B * S, F)
        
        num_vals = X_flat[:, : self.d_numerical]
        num_inv = self.scaler_child.inverse_transform(num_vals)
        num_inv_df = pd.DataFrame(num_inv, columns=self.dt_cols + self.num_cols)

        if self.cat_cols:
            cat_vals = X_flat[:, self.d_numerical:]
            cat_vals = np.round(cat_vals).astype(int)
            cat_df = pd.DataFrame(cat_vals, columns=self.cat_cols)
            cat_inv_df = self.ht.reverse_transform(cat_df)
            result_df = pd.concat([num_inv_df, cat_inv_df], axis=1)
        else:
            result_df = num_inv_df

        result_np = result_df.values.reshape(B, S, -1)
        return result_np


    def inverse_transform_parent(self, df):
        if isinstance(df, torch.Tensor):
            df = df.detach().cpu().numpy()

        is3d = df.ndim == 3
        if is3d:
            B, S, F = df.shape
            X_flat = df.reshape(B * S, F)
        else:
            X_flat = df
            
        num_vals = X_flat[:, : len(self.parent_num)]
        num_inv = self.scaler_parent_parent.inverse_transform(num_vals)
        num_inv_df = pd.DataFrame(num_inv, columns=self.parent_num)

        if self.parent_cat:
            cat_vals = X_flat[:, len(self.parent_num):]
            cat_vals = np.round(cat_vals).astype(int)
            cat_df = pd.DataFrame(cat_vals, columns=self.parent_cat)

            cat_inv_df = self.parent_ht.reverse_transform(cat_df)
            result_df = pd.concat([num_inv_df, cat_inv_df], axis=1)[self.parent_num + self.parent_cat]
        else:
            result_df = num_inv_df

        if is3d:
            return result_df.values.reshape(B, S, -1)
        else:
            return result_df