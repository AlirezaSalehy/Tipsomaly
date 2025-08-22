from .mvtec import MVTEC_CLS_NAMES
from .visa import VISA_CLS_NAMES
from .mpdd import MPDD_CLS_NAMES
from .btad import BTAD_CLS_NAMES
from .sdd import SDD_CLS_NAMES
from .dagm import DAGM_CLS_NAMES
from .dtd import DTD_CLS_NAMES
from .isic import ISIC_CLS_NAMES
from .colondb import ColonDB_CLS_NAMES
from .clinicdb import ClinicDB_CLS_NAMES
from .tn3k import TN3K_CLS_NAMES
from .headct import HEADCT_CLS_NAMES
from .brain_mri import BrainMRI_CLS_NAMES
from .br35h import Br35h_CLS_NAMES
from torch.utils.data import ConcatDataset

dataset_dict = {
    'br35h': (Br35h_CLS_NAMES),
    'brain_mri': (BrainMRI_CLS_NAMES),
    'btad': (BTAD_CLS_NAMES),
    'clinicdb': (ClinicDB_CLS_NAMES),
    'colondb': (ColonDB_CLS_NAMES),
    'dagm': (DAGM_CLS_NAMES),
    'dtd': (DTD_CLS_NAMES),
    'headct': (HEADCT_CLS_NAMES),
    'isic': (ISIC_CLS_NAMES),
    'mpdd': (MPDD_CLS_NAMES),
    'mvtec': (MVTEC_CLS_NAMES),
    'sdd': (SDD_CLS_NAMES),
    'tn3k': (TN3K_CLS_NAMES),
    'visa': (VISA_CLS_NAMES),
}