import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from cnn_arch import NMSELoss,DCE_N,SC_N,Conv_N,FC_N
from synthetic_data import RandomDataset,generate_MMSE_estimate
import matplotlib.pyplot as plt


class model_test():
    def __init__(self):
        super().__init__()
        self.training_SNRdb = 10
        self.num_workers = 8
        self.batch_size = 200
        self.batch_size_DML = 256
        self.data_len = 10000
        self.indicator = -1
        self.data_len_for_test = 10000
        
        # Check GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")


    def test_for_SC(self):
        device = 'cuda'
        Pilot_num2 = 128
        SNRdb = np.arange(5, 16, 2)

        # Load model
        SC2 = SC_N().to(self.device)
        fp = os.path.join(f'./models/HDCE', f'{self.batch_size_DML}_{self.training_SNRdb}dB_epoch9_DML_SC.pth')
        try:
            SC2.load_state_dict(torch.load(fp, map_location=self.device))
        except:
            SC2.load_state_dict(torch.load(fp, map_location=self.device)['cnn'])

        acc2 = []

        with torch.no_grad():
            for i in SNRdb:
                # Setup test data loader
                test_loader2 = DataLoader(
                    dataset=RandomDataset(data_len=self.data_len_for_test, indicator=self.indicator, Pilot_num=Pilot_num2, SNRdb=i),
                    batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False, pin_memory=True)

                print(f"\nSNR: {i} dB")

                pred2_list = []

                label2_list= []


                for Yp, H_label, H_perfect, indicator in test_loader2:
                    bs = Yp.shape[0]
                    # Convert complex to real by concatenating
                    label_out = indicator.long().to(device)
                    Yp_input0 = torch.cat([Yp.real, Yp.imag], dim=1).float().to(device)
                    pred_indicator = SC2(Yp_input0.reshape(bs, 2, 16, 8))
                    pred = pred_indicator.argmax(dim=1)

                    pred2_list.append(pred)
                    label2_list.append(label_out)


                pred2 = torch.cat(pred2_list, dim=0)
                label2 = torch.cat(label2_list, dim=0)
                acc2.append(pred2.eq(label2.view_as(pred2)).sum().item() / (len(label2)))
                print(acc2)

        # === Plotting accuracy vs SNR ===
        plt.figure(figsize=(8, 5))
        plt.plot(SNRdb, acc2, marker='o', linestyle='-', color='b')
        plt.grid(True)
        plt.title('Accuracy vs SNR (dB)')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Accuracy')
        plt.xticks(SNRdb)
        plt.ylim(0, 1)  # accuracy between 0 and 1
        plt.show(block=False)

        return acc2

    def test_for_CE_N(self):
        Pilot_num = 128
        SNRdb = np.arange(5, 16, 2)

        def load_model(model_class, model_path, model_key=None, device=None):
            """Helper function to load models with state dict processing"""
            device = device or self.device
            model = model_class().to(device)
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different model key names in checkpoints
            state_dict = checkpoint[model_key] if model_key else checkpoint
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            model.eval()
            return model

        # Load all models using the helper function
        models = {
            'CNN_scenario0': ('DCE', f'./models/DCE/{self.data_len}_{self.training_SNRdb}dB_best_scenario0.pth', 'cnn'),
            'CNN_scenario1': ('DCE', f'./models/DCE/{self.data_len}_{self.training_SNRdb}dB_best_scenario1.pth', 'cnn'),
            'CNN_scenario2': ('DCE', f'./models/DCE/{self.data_len}_{self.training_SNRdb}dB_best_scenario2.pth', 'cnn'),
            'CNN_DML': ('DCE', f'./models/DCE/{self.batch_size_DML}_{self.training_SNRdb}dB_epoch9_DML.pth', 'cnn'),
            'SC': ('SC', f'./models/HDCE/{self.batch_size_DML}_{self.training_SNRdb}dB_epoch9_DML_SC.pth', 'cnn'),
            'Conv0': ('Conv', f'./models/HDCE/Conv0_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch9_DML.pth', 'conv'),
            'Conv1': ('Conv', f'./models/HDCE/Conv1_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch9_DML.pth', 'conv'),
            'Conv2': ('Conv', f'./models/HDCE/Conv2_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch9_DML.pth', 'conv'),
            'CE': ('FC', f'./models/HDCE/Linear_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch9_DML.pth', 'linear'),
            'CE0': ('DCE', f'./models/HDCE/CE0_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch9_DML.pth', 'ce'),
            'CE1': ('DCE', f'./models/HDCE/CE1_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch9_DML.pth', 'ce'),
            'CE2': ('DCE', f'./models/HDCE/CE2_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch9_DML.pth', 'ce'),
        }

        # Dynamically create model instances based on the models dictionary
        for name, (model_class, path, key) in models.items():
            class_map = {
                'DCE': DCE_N,
                'SC': SC_N,
                'Conv': Conv_N,
                'FC': FC_N
            }
            setattr(self, name, load_model(class_map[model_class], path, key))

        criterion = NMSELoss()
        metrics = {
            'scenario0': [],
            'scenario1': [],
            'scenario2': [],
            'DCE': [],
            'HDCE': [],
            'SDCE': [],
            'LS': [],
            'MMSE': []
        }

        with torch.no_grad():
            for snr in SNRdb:
                test_dataset = RandomDataset(
                    data_len=self.data_len_for_test,
                    indicator=self.indicator,
                    Pilot_num=Pilot_num,
                    SNRdb=snr
                )
                test_loader = DataLoader(
                    dataset=test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    drop_last=False,
                    pin_memory=True
                )

                print('\n')
                print(f'SNR: {snr} dB')

                # Initialize lists for storing results
                result_lists = {
                    'Hhat0': [], 'Hhat1': [], 'Hhat2': [], 'Hhat_DCE': [],
                    'Hperfect': [], 'HLS': [], 'HMMSE': [],
                    'Hhat_HDCE': [], 'Hhat_SDCE': [], 'Hperfect_HDCE': []
                }

                for Yp, HLS, Hperfect, indicator in test_loader:
                    # Generate MMSE estimate
                    HMMSE = generate_MMSE_estimate(HLS.numpy(), sigma2=10**(-snr/10))
                    HMMSE = torch.from_numpy(HMMSE).to(self.device)
                    result_lists['HMMSE'].append(torch.cat([HMMSE.real, HMMSE.imag], dim=1).float().cpu())

                    bs = Yp.shape[0]
                    label_out = torch.cat([HLS.real, HLS.imag], dim=1).float().to(self.device)
                    perfect_out = torch.cat([Hperfect.real, Hperfect.imag], dim=1).float().to(self.device)
                    Yp_input = torch.stack([Yp.real, Yp.imag], dim=1).reshape(bs, 2, 16, 8).float().to(self.device)

                    # Process through all scenario models
                    for i, model in enumerate([self.CNN_scenario0, self.CNN_scenario1, self.CNN_scenario2, self.CNN_DML]):
                        result_lists[f'Hhat{i}' if i < 3 else 'Hhat_DCE'].append(model(Yp_input).cpu())

                    result_lists['HLS'].append(label_out.cpu())
                    result_lists['Hperfect'].append(perfect_out.cpu())

                    # HDCE processing
                    pred_indicator = self.SC(Yp_input)
                    pred = pred_indicator.argmax(dim=1)

                    Yp_class = [[], [], []]
                    label_class = [[], [], []]
                    for i, m in enumerate(pred):
                        Yp_class[m].append(Yp_input[i])
                        label_class[m].append(perfect_out[i])

                    for m in range(3):
                        if Yp_class[m]:
                            hh = torch.stack(label_class[m], dim=0)
                            result_lists['Hperfect_HDCE'].append(hh.cpu())
                            
                            yy = torch.stack(Yp_class[m], dim=0)
                            conv_model = getattr(self, f'Conv{m}')
                            h_out = self.CE(conv_model(yy)).cpu()
                            result_lists['Hhat_HDCE'].append(h_out)
                            
                            ce_model = getattr(self, f'CE{m}')
                            result_lists['Hhat_SDCE'].append(ce_model(yy).cpu())

                # Concatenate all results
                concatenated = {k: torch.cat(v, dim=0) for k, v in result_lists.items() if v}
                
                # Calculate NMSE metrics
                metrics['scenario0'].append(criterion(concatenated['Hhat0'], concatenated['Hperfect']).item())
                metrics['scenario1'].append(criterion(concatenated['Hhat1'], concatenated['Hperfect']).item())
                metrics['scenario2'].append(criterion(concatenated['Hhat2'], concatenated['Hperfect']).item())
                metrics['DCE'].append(criterion(concatenated['Hhat_DCE'], concatenated['Hperfect']).item())
                metrics['HDCE'].append(criterion(concatenated['Hhat_HDCE'], concatenated['Hperfect_HDCE']).item())
                metrics['SDCE'].append(criterion(concatenated['Hhat_SDCE'], concatenated['Hperfect_HDCE']).item())
                metrics['LS'].append(criterion(concatenated['HLS'], concatenated['Hperfect']).item())
                metrics['MMSE'].append(criterion(concatenated['HMMSE'], concatenated['Hperfect']).item())

                # Print current metrics
                for name, values in metrics.items():
                    print(f'NMSE_for_{name}: {values}')

                # Print GPU memory usage if using CUDA
                if self.device.type == 'cuda':
                    print(f"GPU Memory Usage:")
                    print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
                    print(f"Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")

        # Plot results
        plt.figure()
        lines = [
            ('m>-', 'scenario0', 'DCE network only trained on scenario 1'),
            ('b^-', 'scenario1', 'DCE network only trained on scenario 2'),
            ('c<-', 'scenario2', 'DCE network only trained on scenario 3'),
            ('rs-', 'DCE', 'proposed DML based DCE network'),
            ('rd-', 'HDCE', 'proposed DML based HDCE network'),
            ('ro-', 'SDCE', 'DML based SDCE network'),
            ('k--', 'LS', 'LS algorithm'),
            ('r--', 'MMSE', 'MMSE algorithm')
        ]
        
        for style, key, label in lines:
            plt.plot(SNRdb, 10*np.log10(metrics[key]), style, label=label)
        
        plt.grid()
        plt.legend()
        plt.xlabel('SNR (dB)')
        plt.ylabel('NMSE (dB)')
        plt.xticks(np.arange(5, 16, 2))
        plt.yticks(np.arange(-20, 1, 5))
        plt.savefig(f'results/NMSEvsSNR_for_128_10dB_scenario{self.indicator}')
        plt.show(block=False)

        return 0


if __name__ == '__main__':
    print('start to test ...')
    test = model_test()

    print('generate the figure about NMSE performance comparison for the channel scenario 1 with the compression ratio 1/8')
    test.indicator = 0
    test.test_for_CE_N()
    
    print('generate the figure about  NMSE performance comparison for the entire cell with the compression ratio 1/8')
    test.indicator = -1
    test.test_for_CE_N()

    print('generate the figure about the accuracy of the channel scenario prediction')
    test.test_for_SC()