import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
import pandas as pd

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data, DataLoader

import zipfile
import glob
import os

def open_zipfile(path):
  dirs = [i for i in os.listdir(path) if osp.isdir(i)]
  if dirs == []:
    filename = glob.glob(os.path.join(path, "*.zip"))[-1]
    with zipfile.ZipFile(filename, "r") as f:
      f.extractall(path)

class Catalyst(InMemoryDataset):
    r"""
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`QM9` dataset 
        which is from `"Quantum chemistry structures and properties of 134 kilo molecules" <https://www.nature.com/articles/sdata201422>`_ paper.
        It connsists of about 130,000 equilibrium molecules with 12 regression targets: 
        :obj:`mu`, :obj:`alpha`, :obj:`homo`, :obj:`lumo`, :obj:`gap`, :obj:`r2`, :obj:`zpve`, :obj:`U0`, :obj:`U`, :obj:`H`, :obj:`G`, :obj:`Cv`.
        Each molecule includes complete spatial information for the single low energy conformation of the atoms in the molecule.
    
        Args:
            root (string): the dataset folder will be located at root/catalyst.
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)

        Example:
        --------

        >>> dataset = QM93D()
        >>> target = 'mu'
        >>> dataset.data.y = dataset.data[target]
        >>> split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
        >>> train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
        >>> train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        >>> data = next(iter(train_loader))
        >>> data
        Batch(Cv=[32], G=[32], H=[32], U=[32], U0=[32], alpha=[32], batch=[579], gap=[32], homo=[32], lumo=[32], mu=[32], pos=[579, 3], ptr=[33], r2=[32], y=[32], z=[579], zpve=[32])

        Where the attributes of the output data indicates:
    
        * :obj:`z`: The atom type.
        * :obj:`pos`: The 3D position for atoms.
        * :obj:`y`: The target property for the graph (molecule).
        * :obj:`batch`: The assignment vector which maps each node to its respective graph identifier and can help reconstructe single graphs
    """
    def __init__(self, root = 'dataset/', transform = None, pre_transform = None, pre_filter = None):
        self.url = "https://www.dropbox.com/s/46beijsff3aukgm/catalyst_structure.zip?dl=1"
        self.folder = osp.join(root, 'catalyst')
        super(Catalyst, self).__init__(self.folder, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_folder_name(self):
        return 'catalyst structure'
    
    @property
    def raw_file_names(self):
        return 'catalyst_structure.npz'

    @property
    def processed_file_names(self):
         return 'catalyst.pt'

    def download(self):
        download_url(self.url, self.raw_dir)
        open_zipfile(self.raw_dir)


    def process(self):
        #create npz file
        pycatalyst = PyCatalyst(osp.join(self.raw_dir, self.raw_folder_name))
        train_df = pycatalyst.create_df()
        molecules = pycatalyst.create_molecules(train_df)
        npz_dir_path = osp.join(self.raw_dir, self.raw_file_names)
        pycatalyst.create_npz_file(molecules,npz_dir_path)


        data = np.load(osp.join(self.raw_dir, self.raw_file_names),allow_pickle=True)
        R = data['R']
        Z = data['Z']
        N= data['N']
        split = np.cumsum(N)
        R_qm9 = np.split(R, split)
        Z_qm9 = np.split(Z,split)
        target = {}
        for name in ['Pm']:
            target[name] = np.expand_dims(data[name],axis=-1)

        data_list = []
        for i in tqdm(range(len(N))):
            R_i = torch.tensor(R_qm9[i],dtype=torch.float32)
            z_i = torch.tensor(Z_qm9[i],dtype=torch.int64)
            y_i = [torch.tensor(target[name][i],dtype=torch.float32) for name in ['Pm']]

            data = Data(pos=R_i, z=z_i, y=y_i[0], Pm=y_i[0])

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

class Molecule:

  atom_num_dict = {
    'H': 1, 'He':2, 'Li': 3, 'Be':4, 'B': 5,
    'C': 6, 'N': 7, 'O':8, 'F': 9, 'Ne': 10,
    'Na':11, 'Mg':12, 'Al':13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'Br': 35
    }

  def __init__(self,name, pm,id):
    self.id =id
    self.name = name
    self.pm = float(pm)
    self.info_exists=True

  def read_data(self, root):
    inputfile = root+'/'+self.name+'.xyz'
    if os.path.exists(inputfile) == False:
      self.info_exists = False
      return

    df  = pd.read_table(inputfile, skiprows=2, delim_whitespace=True, names=['atom','x', 'y', 'z'])
    df['atom'] = [self.atom_num_dict[atom] for atom in df['atom']]
    xyz = open(inputfile)
    self.N = int(xyz.readline())
    self.R = df[['x','y','z']].values.tolist()
    self.Z = df[['atom']].values.tolist()
    self.Z = [ x[0] for x in self.Z]

  def __str__(self): 
    print("number of atoms: ", self.N)

    print("Positions: ", self.R)
    print("Atoms number", self.Z)
    return ""

class PyCatalyst:

  def __init__(self, path):
    self.path = path
    self.summary_filename = glob.glob(os.path.join(self.path, "*.xlsx"))[-1]
    self.data_filenames = glob.glob(os.path.join(self.path, "*.xyz"))
    
  def create_df(self):
    xlsx_dict = pd.read_excel(self.summary_filename)
    xlsx_dict = pd.DataFrame(xlsx_dict)
   
    train_dict = xlsx_dict[['Ligand' , 'Pm ']]
    test_dict = xlsx_dict[['Ligand.1', 'Pm .1']]
    train_dict.rename(columns={'Pm ':'Pm'} , inplace=True)
    test_dict.rename(columns={'Pm .1': 'Pm' , 'Ligand.1':'Ligand'},inplace=True)
    df = pd.concat([train_dict, test_dict], ignore_index=True)
    df.dropna(inplace=True)
    df = df[df.Pm != '-']

    return df
    

  def create_molecules(self,input_df):
    molecules=[]
    for i in range(len(input_df)):
      molecule_now = Molecule(input_df.iloc[i]['Ligand'], input_df.iloc[i]['Pm'],id=i)
      molecule_now.read_data(self.path)
      if molecule_now.info_exists ==True:
        molecules.append(molecule_now)
    print('\nNumber of molecules: ',len(molecules))
    return  molecules
  

  def create_npz_file(self,molecules,npz_dir):
    R,N,Pm,id, Z = [],[],[],[],[]
    for molecule in molecules:
      R.extend(molecule.R)
      N.append(molecule.N)
      Pm.append(molecule.pm)
      id.append( molecule.id)
      Z.extend(molecule.Z)

    Z, R, N, Pm, id = np.array(Z), np.array(R), np.array(N), np.array(Pm),np.array(id)
    np.savez(npz_dir, Z =Z, R=R,N=N, Pm=Pm,id=id)

if __name__ == '__main__':
    dataset = Catalyst()
    print(dataset)
    print(dataset.data.z.shape)
    print(dataset.data.pos.shape)
    target = 'Pm'
    dataset.data.y = dataset.data[target]
    print(dataset.data.y.shape)
    print(dataset.data.y)
    print(dataset.data.mu)
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
    print(split_idx)
    print(dataset[split_idx['train']])
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    data = next(iter(train_loader))
    print(data)