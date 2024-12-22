from __future__ import print_function, division

import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self, csv_path='dataset_csv/ccrcc_clean.csv', mode='omic', apply_sig=False,
                 shuffle=False, seed=7, print_info=True, n_bins=4, ignore=[],
                 patient_strat=False, label_col=None, filter_dict={}, eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        slide_data = pd.read_csv(csv_path, low_memory=False)
        ### new
        missing_slides_ls = ['TCGA-A7-A6VX-01Z-00-DX2.9EE94B59-6A2C-4507-AA4F-DC6402F2B74F.svs',
                             'TCGA-A7-A0CD-01Z-00-DX2.609CED8D-5947-4753-A75B-73A8343B47EC.svs',
                             'TCGA-HT-7483-01Z-00-DX1.7241DF0C-1881-4366-8DD9-11BF8BDD6FBF.svs',
                             'TCGA-06-0882-01Z-00-DX2.7ad706e3-002e-4e29-88a9-18953ba422bf.svs',
                             # TCGA-LUAD
                             'TCGA-49-4505-01Z-00-DX4.623c4278-fc3e-4c80-bb4d-000e24fbb1c2.svs',  # [77775, 1024]
                             'TCGA-49-4505-01Z-00-DX3.d3f8d0c4-e4bc-4201-9b2f-e010f51dd8c3.svs',  # [61419, 1024]
                             'TCGA-49-4506-01Z-00-DX4.9460a2c1-efe2-4095-9b0e-dfe0eda684ba.svs',  # [87632, 1024]
                             'TCGA-49-4506-01Z-00-DX3.a05a3969-bbef-48d8-86de-f51c8870afd6.svs',  # [65332, 1024]
                             'TCGA-49-6745-01Z-00-DX7.4bdde497-03d9-4e15-a00e-595baa3947b5.svs',  # [191335, 1024]
                             'TCGA-49-6745-01Z-00-DX6.3174fffa-c17c-44d1-8d21-1921a9ec1bee.svs',  # [170859, 1024]
                             'TCGA-49-6745-01Z-00-DX5.9521a16f-d6a7-460c-b75c-7341d72ce727.svs',  # [153881, 1024]
                             'TCGA-49-6745-01Z-00-DX4.48b4bbb5-4214-4bfb-91e8-bddfa2f03c92.svs',  # [131939, 1024]
                             'TCGA-49-6745-01Z-00-DX3.40cd3c60-889c-4eaa-be55-36ab5d8b2400.svs',  # [91800, 1024]
                             'TCGA-49-6745-01Z-00-DX2.3b8e947b-885e-486f-9edc-f2b247bdf95b.svs',  # [57759, 1024]
                             'TCGA-49-4488-01Z-00-DX9.bf1e9d28-3ca2-4a62-bf3e-3e735c6a2885.svs',  # [167497, 1024]
                             'TCGA-49-4488-01Z-00-DX8.b9fb358a-9618-4956-9544-c2ffe2678095.svs',  # [146438, 1024]
                             'TCGA-49-4488-01Z-00-DX7.a04b36c3-9cfd-4195-92c0-70046cd49d9a.svs',  # [124001, 1024]
                             'TCGA-49-4488-01Z-00-DX6.3ce306b2-1b00-48f7-8288-c24069037a9c.svs',  # [115154, 1024]
                             'TCGA-49-4488-01Z-00-DX5.6380c697-c81b-48c8-863c-ea1cdd097464.svs',  # [106087, 1024]
                             'TCGA-49-4488-01Z-00-DX4.17c00349-5908-4944-9dba-5efea9a5373a.svs',  # [86268, 1024]
                             'TCGA-49-4488-01Z-00-DX3.341ea7ac-ee5f-4ad2-9d01-d81a4d2e9b13.svs',  # [70964, 1024]
                             'TCGA-49-4490-01Z-00-DX6.0a04ad8e-7f33-4496-bd9e-0110ec33f887.svs',  # [119750, 1024]
                             'TCGA-49-4490-01Z-00-DX5.feea87ff-61c6-465e-8b36-4803f6af3d62.svs',  # [90674, 1024]
                             'TCGA-49-4490-01Z-00-DX4.4e94aeb6-f616-4bdd-82bc-2ca2f2b5a5c4.svs',  # [70737, 1024]
                             'TCGA-49-4490-01Z-00-DX3.0949a10a-0a1f-49dc-9296-ab4d4f0bf988.svs',  # [52322, 1024]
                             'TCGA-49-4501-01Z-00-DX3.b6c2cc84-1c94-4816-92e7-8cf4446ac9ac.svs',  # [73729, 1024]
                             'TCGA-49-4501-01Z-00-DX2.07b7723d-136b-490c-9dd3-7f3aef78d3d9.svs',  # [55763, 1024]
                             'TCGA-49-4514-01Z-00-DX4.79e9e14e-72ea-4e9f-9719-d1a791cf0b43.svs',  # [65936, 1024]
                             'TCGA-49-4494-01Z-00-DX7.9f722476-622e-4374-9fda-ef35184f01e0.svs',  # [145832, 1024]
                             'TCGA-49-4494-01Z-00-DX6.3791a027-7ddb-429c-99a5-bb84a4307550.svs',  # [125025, 1024]
                             'TCGA-49-4494-01Z-00-DX5.1e9e22d6-a4c9-40d1-aedb-b6cd404fe16f.svs',  # [110212, 1024]
                             'TCGA-49-4494-01Z-00-DX4.4c6a4560-adee-4fc9-8fd8-feb7f89e51e3.svs',  # [90559, 1024]
                             'TCGA-49-4494-01Z-00-DX3.e80b4534-4d6e-4d79-962e-017ffee24d67.svs',  # [69059, 1024]
                             'TCGA-49-4494-01Z-00-DX2.cac5ed0a-98c3-4d37-a4f4-9596a061836a.svs',  # [52863, 1024]
                             'TCGA-50-5068-01Z-00-DX2.0492A5C6-09CB-424B-BE20-10A1CBEA2E57.svs',  # [51576, 1024]
                             'TCGA-49-4512-01Z-00-DX8.c54237a3-93f4-432f-90f1-b930b7341741.svs',  # [164110, 1024]
                             'TCGA-49-4512-01Z-00-DX7.1f758560-85e2-4cf9-a81a-317536d96bcc.svs',  # [148722, 1024]
                             'TCGA-49-4512-01Z-00-DX6.985c9088-ff83-4f13-8a95-cb55aa48682b.svs',  # [132007, 1024]
                             'TCGA-49-4512-01Z-00-DX5.7198ce36-1fae-4da1-9f26-b7f43cf01133.svs',  # [111569, 1024]
                             'TCGA-49-4512-01Z-00-DX4.5161fcf5-62a3-4f7d-8a55-9a24a7d70efb.svs',  # [80672, 1024]
                             'TCGA-49-4512-01Z-00-DX3.2f6ec7bc-0dac-4be0-95fe-4071a93a856f.svs',  # [62186, 1024]
                             'TCGA-49-6744-01Z-00-DX4.a3d7995d-399f-4c53-aab8-adc4ea4dbfa8.svs',  # [107325, 1024]
                             'TCGA-49-6744-01Z-00-DX3.fa64d744-6caa-4f62-8689-33da9c312838.svs',  # [87156, 1024]
                             'TCGA-49-6742-01Z-00-DX5.74539ee6-0ac6-4663-89f1-c6a74a59a4cb.svs',  # [155787, 1024]
                             'TCGA-49-6742-01Z-00-DX4.a11201e1-9eeb-40ea-9c88-28de847ef7d8.svs',  # [122612, 1024]
                             'TCGA-49-6742-01Z-00-DX3.789aa445-e63f-4e6e-900a-0f47f44af6b7.svs',  # [89779, 1024]
                             'TCGA-49-6742-01Z-00-DX2.2c6b4df0-867d-40c5-8bee-14e2d219224b.svs',  # [61791, 1024]
                             'TCGA-49-6743-01Z-00-DX4.5ea81023-1372-48ed-88a8-374e2394d7f7.svs',  # [95344, 1024]
                             'TCGA-49-6743-01Z-00-DX3.d391f3ab-e954-4f4f-996d-cd0420933ccf.svs',  # [82289, 1024]
                             'TCGA-73-4658-01Z-00-DX1.d5beb44f-9d76-485a-8af4-407b0f1a610e.svs',  # [48475, 1024]
                             'TCGA-49-6744-01Z-00-DX2.1982e585-65a4-4330-9140-ccabcdd106f8.svs',  # [45030, 1024]
                             'TCGA-49-4488-01Z-00-DX2.ce896a98-8b45-4606-8849-3a377b81e3de.svs',  # [46359, 1024]
                             'TCGA-50-5055-01Z-00-DX2.446EC3BB-2ED7-4253-8A39-AC68331F08E7.svs',  # [45859, 1024]
                             'TCGA-49-4514-01Z-00-DX3.3b247c08-3069-4100-9df4-734895adb954.svs',  # [46187, 1024]
                             'TCGA-49-4506-01Z-00-DX2.81e180ae-bd9f-49cc-9a0f-297800cc7280.svs',  # [44241, 1024]
                             'TCGA-J2-8192-01Z-00-DX1.A784F381-7906-480F-99A1-0B88005953A0.svs',  # [46565, 1024]
                             'TCGA-MN-A4N1-01Z-00-DX2.9B0852C4-16BF-4962-B86F-E2570E48A89E.svs',  # [48475, 1024]
                             'TCGA-49-6743-01Z-00-DX2.f6b71e89-19ff-4d9b-a3f1-3a52949f1dc7.svs',  # [55361, 1024]
                             ]
        slide_data.drop(slide_data[slide_data['slide_id'].isin(missing_slides_ls)].index, inplace=True)

        # slide_data = slide_data.drop(['Unnamed: 0'], axis=1)
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)

        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        if "IDC" in slide_data['oncotree_code']:  # must be BRCA (and if so, use only IDCs)
            slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps

        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False,
                                     include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient: slide_ids})

        self.patient_dict = patient_dict

        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins) - 1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c): key_count})
                key_count += 1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes = len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id': patients_df['case_id'].values, 'label': patients_df['label'].values}

        new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:12]
        self.mode = mode
        self.cls_ids_prep()

        ### Signatures
        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('./datasets_csv_sig/signatures.csv')
        else:
            self.signatures = None

        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id']))  # get unique patients
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]]  # get patient label
            patient_labels.append(label)

        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort=False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    def get_split_from_df(self, all_splits: dict, split_key: str = 'train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata, mode=self.mode,
                                  signatures=self.signatures, data_dir=self.data_dir,
                                  label_col=self.label_col, patient_dict=self.patient_dict,
                                  num_classes=self.num_classes)
        else:
            split = None

        return split

    def return_splits(self, from_id: bool = True, csv_path: str = None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
            test_split = None  # self.get_split_from_df(all_splits=all_splits, split_key='test')

            ### --> Normalizing Data
            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)
            # test_split.apply_scaler(scalers=scalers)
            ### <--
        return train_split, val_split  # , test_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def __getitem__(self, idx):
        return None


class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, mode: str = 'omic', **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        c = self.slide_data['censorship'][idx]
        slide_ids = self.patient_dict[case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not self.use_h5:
            if self.data_dir:
                if self.mode == 'path':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path, map_location=torch.device('cpu'))
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    return (path_features, torch.zeros((1, 1)), label, event_time, c)

                elif self.mode == 'cluster':
                    path_features = []
                    cluster_ids = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path, map_location=torch.device('cpu'))
                        path_features.append(wsi_bag)
                        cluster_ids.extend(self.fname2ids[slide_id[:-4] + '.pt'])
                    path_features = torch.cat(path_features, dim=0)
                    cluster_ids = torch.Tensor(cluster_ids)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, cluster_ids, genomic_features, label, event_time, c)

                elif self.mode == 'omic':
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (torch.zeros((1, 1)), genomic_features, label, event_time, c)

                elif self.mode == 'pathomic':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path, map_location=torch.device('cpu'))
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, genomic_features, label, event_time, c)

                elif self.mode == 'coattn':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        # wsi_bag = torch.load(wsi_path,map_location=torch.device('cpu'))
                        wsi_bag = torch.load(wsi_path, map_location=torch.device('cpu'), weights_only=True)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx].values)
                    omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx].values)
                    omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx].values)
                    omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx].values)
                    omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx].values)
                    omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx].values)
                    return (path_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c, slide_id)
                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
                ### <--
            else:
                return slide_ids, label, event_time, c


class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata, mode, signatures=None, data_dir=None, label_col=None, patient_dict=None,
                 num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        ### --> Initializing genomic features in Generic Split
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        self.signatures = signatures

        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))

        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                omic = np.concatenate([omic + mode for mode in ['_mut', '_cnv', '_rnaseq']])
                omic = sorted(series_intersection(omic, self.genomic_features.columns))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]
        # print("Shape", self.genomic_features.shape)
        ### <--

    def __len__(self):
        return len(self.slide_data)

    ### --> Getting StandardScaler of self.genomic_features
    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)

    ### <--

    ### --> Applying StandardScaler to self.genomic_features
    def apply_scaler(self, scalers: tuple = None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed
    ### <--
