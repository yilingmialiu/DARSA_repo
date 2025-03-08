import torch.nn as nn
import sys
sys.path.insert(0, '../')
from utils.helperDSN import *


class DSN(nn.Module):
    def __init__(self, code_size=100, n_class=3):
        super(DSN, self).__init__()
        self.code_size = code_size

        ##########################################
        # private source encoder
        ##########################################

        self.source_encoder_conv = nn.Sequential(
            nn.Linear(616, 256), 
            nn.ReLU(),
            nn.Linear(256, code_size), 
        )

        #########################################
        # private target encoder
        #########################################

        self.target_encoder_conv = nn.Sequential(
            nn.Linear(616, 256), 
            nn.ReLU(),
            nn.Linear(256, code_size), 
        )

        ################################
        # shared encoder (dann_mnist)
        ################################

        self.shared_encoder_conv = nn.Sequential(
            nn.Linear(616, 256), 
            nn.ReLU(),
            nn.Linear(256, code_size), 
        )

        # classify 10 numbers
        self.shared_encoder_pred_class = nn.Sequential()
        self.shared_encoder_pred_class.add_module('fc_se4', nn.Linear(in_features=code_size, out_features=100))
        self.shared_encoder_pred_class.add_module('relu_se4', nn.ReLU(True))
        self.shared_encoder_pred_class.add_module('fc_se5', nn.Linear(in_features=100, out_features=n_class))

        self.shared_encoder_pred_domain = nn.Sequential()
        self.shared_encoder_pred_domain.add_module('fc_se6', nn.Linear(in_features=100, out_features=100))
        self.shared_encoder_pred_domain.add_module('relu_se6', nn.ReLU(True))

        # classify two domain
        self.shared_encoder_pred_domain.add_module('fc_se7', nn.Linear(in_features=100, out_features=2))

        ######################################
        # shared decoder (small decoder)
        ######################################

        self.shared_decoder_fc = nn.Sequential(
            nn.Linear(code_size, 256), 
            nn.ReLU(),
            nn.Linear(256, 616), 
        )
        

    def forward(self, input_data, mode, rec_scheme, p=0.0):

        result = []

        if mode == 'source':

            # source private encoder
            private_code = self.source_encoder_conv(input_data)

        elif mode == 'target':

            # target private encoder
            private_code = self.target_encoder_conv(input_data)

        result.append(private_code)

        # shared encoder
        shared_code = self.shared_encoder_conv(input_data)
        result.append(shared_code)

        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        result.append(domain_label)

        if mode == 'source':
            class_label = self.shared_encoder_pred_class(shared_code)
            result.append(class_label)

        # shared decoder

        if rec_scheme == 'share':
            union_code = shared_code
        elif rec_scheme == 'all':
            union_code = private_code + shared_code
        elif rec_scheme == 'private':
            union_code = private_code

        rec_code = self.shared_decoder_fc(union_code)
        result.append(rec_code)

        return result





