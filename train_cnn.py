
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os

from cnn_arch import  NMSELoss,DCE_N,SC_N,Conv_N,FC_N
from synthetic_data import DatasetFolder,DatasetFolder_DML


class Model_train():
    def __init__(self):
        self.Pilot_num = 128
        self.data_len = 10000   
        self.SNRdb = 10        
        self.num_workers = 4    
        self.batch_size = 256    
        self.batch_size_DML = 256
        self.lr = 1e-3
        self.lr_decay = 30
        self.lr_threshold = 1e-6
        self.n_epochs = 10       
        self.print_freq = 5     
        self.optimizer = 'adam'
        self.train_test_ratio = 0.9

    def get_data(self,data_len,indicator,uid):

        Yp = np.load('training_data/Yp'+str(indicator)+'_' + str(self.Pilot_num) + '_1024_' + str(self.SNRdb) + 'dB_' + str(uid) +'_datalen_'+str(data_len)+'.npy')
        Hlabel = np.load('training_data/Hlabel'+str(indicator)+'_' + str(self.Pilot_num) + '_1024_' + str(self.SNRdb) + 'dB_' + str(uid) + '_datalen_'+str(data_len)+'.npy')
        Hperf = np.load('training_data/Hperf'+str(indicator)+'_' + str(self.Pilot_num) + '_1024_' + str(self.SNRdb) + 'dB_' + str(uid) + '_datalen_'+str(data_len)+'.npy')

        print('data loaded for scenario'+str(indicator)+' user'+str(uid)+'!')
        Indicator = []
        for i in range(data_len):
            Indicator.append(indicator)
        Indicator = np.stack(Indicator, axis=0)

        Yp = Yp[:data_len]
        Hlabel = Hlabel[:data_len]
        Hperf = Hperf[:data_len]

        start = int(Yp.shape[0] * self.train_test_ratio)
        Yp_train, Yp_val = Yp[:start], Yp[start:]
        Hlabel_train, Hlabel_val = Hlabel[:start], Hlabel[start:]
        Hperf_train, Hperf_val = Hperf[:start],Hperf[start:]
        Indicator_train, Indicator_val = Indicator[:start], Indicator[start:]

        return [Yp_train, Hlabel_train, Hperf_train, Indicator_train],[Yp_val,Hlabel_val,Hperf_val,Indicator_val]

    def get_dataloader_DML(self, data_len):
        # get_data returns 4 items: [Yp, Hlabel, Hperf, Indicator]
        td00, vd00 = self.get_data(data_len, 0, 0)
        td01, vd01 = self.get_data(data_len, 0, 1)
        td02, vd02 = self.get_data(data_len, 0, 2)
        td10, vd10 = self.get_data(data_len, 1, 0)
        td11, vd11 = self.get_data(data_len, 1, 1)
        td12, vd12 = self.get_data(data_len, 1, 2)
        td20, vd20 = self.get_data(data_len, 2, 0)
        td21, vd21 = self.get_data(data_len, 2, 1)
        td22, vd22 = self.get_data(data_len, 2, 2)

        train_dataset = DatasetFolder_DML(td00, td01, td02, td10, td11, td12, td20, td21, td22)
        val_dataset = DatasetFolder_DML(vd00, vd01, vd02, vd10, vd11, vd12, vd20, vd21, vd22)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader



    def get_loss(self, td, CNN, criterion, device):

        Yp = td[0]
        Hlabel = td[1]
        Hperfect = td[2]
        bs = len(Yp)

        label_out = torch.cat([Hlabel.real, Hlabel.imag], dim=1).float().to(device)
        perfect_out = torch.cat([Hperfect.real, Hperfect.imag], dim=1).float().to(device)
        Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).float().to(device)
        Hhat = CNN(Yp_input.reshape(bs, 2, 16, 8))

        loss = criterion(Hhat, label_out)
        loss_perf = criterion(Hhat, perfect_out)

        return loss, loss_perf

    def get_estimate(self, vd, CNN, device):

        Yp = vd[0]
        Hlabel = vd[1]
        Hperfect = vd[2]

        bs = Yp.shape[0]
        # complex to real by concatenation
        label_out = torch.cat([Hlabel.real, Hlabel.imag], dim=1).float()
        perfect_out = torch.cat([Hperfect.real, Hperfect.imag], dim=1).float()
        # the input and the output
        Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).to(device)
        Hhat = CNN(Yp_input.reshape(bs, 2, 16, 8).float()).detach().cpu()

        return Hhat,label_out, perfect_out

    def train_DCE_for_scenario(self, scenario_id):
        device = 'cuda'
        CNN = DCE_N().to(device)
        data_len = self.data_len
        td, vd = self.get_data(self.data_len, scenario_id, 0)
        train_loader = torch.utils.data.DataLoader(DatasetFolder(td), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(DatasetFolder(vd), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        print('Data Loaded!')
        
        criterion = NMSELoss()
        optimizer = optim.Adam(CNN.parameters(), self.lr)
        best_nmse = float('inf')
        save_dir = './models/DCE'
        os.makedirs(save_dir, exist_ok=True)
        print('Staring training')
        
        for epoch in range(self.n_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Scenario {scenario_id}: SNR: {self.SNRdb} Epoch [{epoch}/{self.n_epochs}] LR: {current_lr:.4e}', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            
            CNN.train()
            for it, batch in enumerate(train_loader):
                optimizer.zero_grad()
                loss, loss_perf = self.get_loss(batch, CNN, criterion, device)
                loss.backward()
                optimizer.step()
                if it % self.print_freq == 0:
                    print(f'Epoch [{epoch}/{self.n_epochs}] Iter [{it}/{len(train_loader)}] Loss: {loss.item():.5f} Loss_perf: {loss_perf.item():.5f}')
            
            CNN.eval()
            Hhat_list, Hlabel_list, Hperfect_list = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    Hhat, label_out, perfect_out = self.get_estimate(batch, CNN, device)
                    Hhat_list.append(Hhat)
                    Hlabel_list.append(label_out)
                    Hperfect_list.append(perfect_out)
            Hhat = torch.cat(Hhat_list, dim=0)
            Hlabel = torch.cat(Hlabel_list, dim=0)
            Hperfect = torch.cat(Hperfect_list, dim=0)
            
            nmse = criterion(Hhat, Hlabel)
            nmse_perf = criterion(Hhat, Hperfect)
            
            if nmse < best_nmse:
                torch.save({'cnn': CNN.state_dict()}, os.path.join(save_dir, f'{data_len}_{self.SNRdb}dB_best_scenario{scenario_id}.pth'))
                best_nmse = nmse.item()
                print('CNN saved!')
            
            print(f'Epoch [{epoch}/{self.n_epochs}] || NMSE: {nmse.item():.5f}, NMSE_perf: {nmse_perf.item():.5f}, Best NMSE: {best_nmse:.5f}')
            print('=' * 60)
            
            if epoch > 0 and epoch % self.lr_decay == 0:
                optimizer.param_groups[0]['lr'] = max(current_lr * 0.5, self.lr_threshold)


    def train_DCE_for_DML(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        CNN = DCE_N()
        CNN = CNN.to(device)
        print(f"Model is on device: {next(CNN.parameters()).device}")
        
        data_len = self.data_len
        train_loader, val_loader = self.get_dataloader_DML(data_len=data_len)
        print('Data Loaded!')
        criterion = NMSELoss()
        optimizer = optim.Adam(CNN.parameters(), self.lr)
        best_nmse = 1000.
        print('Everything prepared well, start to train...')
        for epoch in range(self.n_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            print('DML:'f'SNR: {self.SNRdb} ' f'Epoch [{epoch}]/[{self.n_epochs}] learning rate: {current_lr:.4e}',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            CNN.train()
            for it, (td00,td01,td02,td10,td11,td12,td20,td21,td22) in enumerate(train_loader):
                sutd=[[td00,td01,td02],[td10,td11,td12],[td20,td21,td22]]
                total_loss=0
                total_loss_perf=0

                optimizer.zero_grad()

                for sid in range(3):
                    for uid in range(3):
                        loss,loss_perf=self.get_loss(sutd[sid][uid],CNN,criterion,device)
                        loss = loss/9
                        loss_perf = loss_perf/9
                        total_loss+=loss
                        total_loss_perf+=loss_perf

                        # calculate gradient
                        loss.backward()
                # update weights
                optimizer.step()
                if it % self.print_freq == 0:
                    print(f'Epoch: [{epoch}/{self.n_epochs}][{it}/{len(train_loader)}]\t Loss {total_loss.item():.5f}\t Loss_perf {total_loss_perf.item():.5f}')

            CNN.eval()
            with torch.no_grad():
                Hhat_list = []
                Hlabel_list = []
                Hperfect_list = []

                for vd00,vd01,vd02,vd10,vd11,vd12,vd20,vd21,vd22 in val_loader:

                    suvd = [[vd00, vd01, vd02], [vd10, vd11, vd12], [vd20, vd21, vd22]]

                    for sid in range(3):
                        for uid in range(3):
                            Hhat, label_out, perfect_out = self.get_estimate(suvd[sid][uid], CNN, device)
                            Hhat_list.append(Hhat)
                            Hlabel_list.append(label_out)
                            Hperfect_list.append(perfect_out)

                Hhat = torch.cat(Hhat_list, dim=0)
                Hlabel = torch.cat(Hlabel_list, dim=0)
                Hperfect = torch.cat(Hperfect_list, dim=0)

                nmse = criterion(Hhat, Hlabel)
                nmse_perf = criterion(Hhat,Hperfect)
                if epoch==self.n_epochs-1:
                    fp = os.path.join(f'./models/DCE',
                                      f'{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth')
                    torch.save({'cnn': CNN.state_dict()}, fp)
                    print('CNN finally saved!')

                if nmse < best_nmse:
                    torch.save({'cnn': CNN.state_dict()}, os.path.join(f'./models/DCE', f'{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))
                    best_nmse = nmse.item()
                    print('CNN saved!')
                print(
                    f'Epoch [{epoch}]/[{self.n_epochs}] || NMSE {nmse.item():.5f},NMSE_perf {nmse_perf.item():.5f}, best nmse: {best_nmse:.5f}')
                print('==============================================================')

            if epoch > 0:
                if epoch % self.lr_decay == 0:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
                if optimizer.param_groups[0]['lr'] < self.lr_threshold:
                    optimizer.param_groups[0]['lr'] = self.lr_threshold
    
    def get_HDCE_loss(self, td, Conv, CE, criterion, device):
        Yp = td[0]
        Hlabel = td[1]
        Hperfect = td[2]
        bs = len(Yp)
        # Convert complex values to real by concatenating real and imaginary parts
        label_out = torch.cat([Hlabel.real, Hlabel.imag], dim=1).float().to(device)
        perfect_out = torch.cat([Hperfect.real, Hperfect.imag], dim=1).float().to(device)
        Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).reshape(bs, 2, 16, 8).float().to(device)
        h_out = Conv(Yp_input)
        Hhat = CE(h_out)

        # Calculate loss with respect to the label and perfect output
        loss = criterion(Hhat, label_out)  # Loss for the channel estimate
        loss_perf = criterion(Hhat, perfect_out)  # Loss for the perfect estimate

        return loss, loss_perf

    def get_HDCE_estimate(self, vd, Conv, CE, device):
        Yp = vd[0]
        Hlabel = vd[1]
        Hperfect = vd[2] 
        bs = len(Yp)
        
        # Convert complex values to real by concatenating real and imaginary parts
        label_out = torch.cat([Hlabel.real, Hlabel.imag], dim=1).float().to(device)
        perfect_out = torch.cat([Hperfect.real, Hperfect.imag], dim=1).float().to(device)
        Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).reshape(bs, 2, 16, 8).float().to(device)
        h_out = Conv(Yp_input)
        # Channel estimation output
        Hhat = CE(h_out)
        # Return the estimated channel, label output, and perfect output
        return Hhat, label_out, perfect_out


    def train_Conv_Linear_of_HDCE(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Initialize the convolutional networks and fully connected network
        Conv0, Conv1, Conv2 = Conv_N(), Conv_N(), Conv_N()
        CE = FC_N()
        Conv0, Conv1, Conv2, CE = Conv0.to(device), Conv1.to(device), Conv2.to(device), CE.to(device)
        data_len = self.data_len
        train_loader, val_loader = self.get_dataloader_DML(data_len=data_len)
        print('Data Loaded!')
        # Loss function and optimizers for each network component
        criterion = NMSELoss()
        optimizers = [
            optim.Adam(Conv0.parameters(), self.lr),
            optim.Adam(Conv1.parameters(), self.lr),
            optim.Adam(Conv2.parameters(), self.lr),
            optim.Adam(CE.parameters(), self.lr)
        ]
        best_nmse = float('inf')
        print('Starting Training.')
        for epoch in range(self.n_epochs):
            current_lr = optimizers[0].param_groups[0]['lr']
            print(f'Conv+Linear of HDCE: SNR: {self.SNRdb} Epoch [{epoch}/{self.n_epochs}] learning rate: {current_lr:.4e}', 
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            Conv0.train(), Conv1.train(), Conv2.train(), CE.train()
            Conv = [Conv0, Conv1, Conv2]
            
            for it, (td00, td01, td02, td10, td11, td12, td20, td21, td22) in enumerate(train_loader):
                sutd = [[td00, td01, td02], [td10, td11, td12], [td20, td21, td22]]
                
                for optimizer in optimizers:
                    optimizer.zero_grad()
                    
                total_loss, total_loss_perf = 0, 0
                
                for sid in range(3):
                    for uid in range(3):
                        loss, loss_perf = self.get_HDCE_loss(sutd[sid][uid], Conv[sid], CE, criterion, device)
                        total_loss += loss / 9
                        total_loss_perf += loss_perf / 9
                total_loss.backward()

                for optimizer in optimizers:
                    optimizer.step()

                if it % self.print_freq == 0:
                    print(f'Epoch: [{epoch}/{self.n_epochs}][{it}/{len(train_loader)}] Loss {total_loss.item():.5f} Loss_perf {total_loss_perf.item():.5f}')

            Conv0.eval(), Conv1.eval(), Conv2.eval(), CE.eval()
            Hhat_list, Hlabel_list, Hperfect_list = [], [], []

            with torch.no_grad():
                for vd00, vd01, vd02, vd10, vd11, vd12, vd20, vd21, vd22 in val_loader:
                    suvd = [[vd00, vd01, vd02], [vd10, vd11, vd12], [vd20, vd21, vd22]]
                    for sid in range(3):
                        for uid in range(3):
                            Hhat, label_out, perfect_out = self.get_HDCE_estimate(suvd[sid][uid], Conv[sid], CE, device)
                    
                            Hhat_list.append(Hhat)
                            Hlabel_list.append(label_out)
                            Hperfect_list.append(perfect_out)

                # Concatenate results
                Hhat = torch.cat(Hhat_list, dim=0)
                Hlabel = torch.cat(Hlabel_list, dim=0)
                Hperfect = torch.cat(Hperfect_list, dim=0)

                # Calculate NMSE for both estimated and perfect values
                nmse = criterion(Hhat, Hlabel)
                nmse_perf = criterion(Hhat, Hperfect)

                # Save models at the final epoch
                if epoch == self.n_epochs - 1:
                    for idx, conv in enumerate([Conv0, Conv1, Conv2]):
                        torch.save({'conv': conv.state_dict()},
                                os.path.join(f'./models/HDCE', f'Conv{idx}_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))
                    torch.save({'linear': CE.state_dict()},
                            os.path.join(f'./models/HDCE', f'Linear_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))
                    print('HDCE finally saved!')

                # Save the best model based on NMSE
                if nmse < best_nmse:
                    for idx, conv in enumerate([Conv0, Conv1, Conv2]):
                        torch.save({'conv': conv.state_dict()},
                                os.path.join(f'./models/HDCE', f'Conv{idx}_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))
                    torch.save({'linear': CE.state_dict()},
                            os.path.join(f'./models/HDCE', f'Linear_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))
                    best_nmse = nmse.item()
                    print('HDCE saved!')

                # Display NMSE information
                print(f'Epoch [{epoch}/{self.n_epochs}] || NMSE {nmse.item():.5f}, NMSE_perf {nmse_perf.item():.5f}, best nmse: {best_nmse:.5f}')

            # Learning rate decay
            if epoch > 0 and epoch % self.lr_decay == 0:
                for optimizer in optimizers:
                    optimizer.param_groups[0]['lr'] *= 0.5
                    if optimizer.param_groups[0]['lr'] < self.lr_threshold:
                        optimizer.param_groups[0]['lr'] = self.lr_threshold

    def get_SE_loss(self, td, CNN, device):
        Yp = td[0]
        indicator=td[3]
        bs = len(Yp)
        label_out = indicator.long().to(device)
        Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).float().to(device)
        pred_indicator = CNN(Yp_input.reshape(bs, 2, 16, 8))
        loss = F.nll_loss(pred_indicator, label_out)
        return loss

    def get_SE_estimate(self, td, CNN, device):
        Yp = td[0]
        indicator = td[3]
        bs = len(Yp)
        label_out = indicator.long().to(device)
        Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).float().to(device)
        pred_indicator = CNN(Yp_input.reshape(bs, 2, 16, 8))
        return pred_indicator,label_out

    def train_SC_of_HDCE(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        CNN = SC_N()
        CNN.to(device)
        data_len = self.data_len
        train_loader, val_loader = self.get_dataloader_DML(data_len=data_len)
        print('Data Loaded!')

        optimizer = optim.Adam(CNN.parameters(), self.lr)
        best_acc = 0

        print('Everything prepared well, start to train...')

        for epoch in range(self.n_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            print('SC of HDCE: 'f'SNR: {self.SNRdb} ' f'Epoch [{epoch}]/[{self.n_epochs}] learning rate: {current_lr:.4e}',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            CNN.train()
            for it, (td00,td01,td02,td10,td11,td12,td20,td21,td22) in enumerate(train_loader):

                sutd = [[td00, td01, td02], [td10, td11, td12], [td20, td21, td22]]
                optimizer.zero_grad()
                total_loss=0

                for sid in range(3):
                    for uid in range(3):
                        loss=self.get_SE_loss(sutd[sid][uid],CNN,device)
                        loss=loss/9
                        total_loss+=loss
                total_loss.backward()

                optimizer.step()
                if it % self.print_freq == 0:

                    print(f'Epoch: [{epoch}/{self.n_epochs}][{it}/{len(train_loader)}]\t Loss {total_loss.item():.5f}')


            CNN.eval()
            with torch.no_grad():
                pred_list = []
                label_list = []

                for vd00,vd01,vd02,vd10,vd11,vd12,vd20,vd21,vd22 in val_loader:
                    suvd = [[vd00, vd01, vd02], [vd10, vd11, vd12], [vd20, vd21, vd22]]

                    for sid in range(3):
                        for uid in range(3):
                            pred_indicator,label_out = self.get_SE_estimate(suvd[sid][uid],CNN,device)
                            pred = pred_indicator.argmax(dim=1)
                            pred_list.append(pred)
                            label_list.append(label_out)

                pred = torch.cat(pred_list, dim=0)
                label = torch.cat(label_list, dim=0)
                acc = pred.eq(label.view_as(pred)).sum().item()/(len(label))

                if epoch==self.n_epochs-1:
                    fp = os.path.join(f'./models/HDCE', f'{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML_SC.pth')
                    torch.save({'cnn': CNN.state_dict()}, fp)
                    print('SC finally saved!')

                if acc > best_acc:
                    torch.save({'cnn': CNN.state_dict()}, os.path.join(f'./models/HDCE',f'{self.batch_size_DML}_{self.SNRdb}dB_best_DML_SC.pth'))
                    best_acc = acc
                    print('SC saved!')
                print(
                    f'Epoch [{epoch}]/[{self.n_epochs}] || acc {acc:.2%}, best acc: {best_acc:.2%}')
            if epoch > 0:
                if epoch % self.lr_decay == 0:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
                if optimizer.param_groups[0]['lr'] < self.lr_threshold:
                    optimizer.param_groups[0]['lr'] = self.lr_threshold

    def train_SDCE(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        CE0=DCE_N()
        CE1=DCE_N()
        CE2=DCE_N()
        CE0.to(device)
        CE1.to(device)
        CE2.to(device)

        data_len = self.data_len
        train_loader, val_loader = self.get_dataloader_DML(data_len=data_len)
        print('Data Loaded!')
        criterion = NMSELoss()
        optimizer_CE0 = optim.Adam(CE0.parameters(), self.lr)
        optimizer_CE1 = optim.Adam(CE1.parameters(), self.lr)
        optimizer_CE2 = optim.Adam(CE2.parameters(), self.lr)

        best_nmse = 1000.

        print('Starting training')
        for epoch in range(self.n_epochs):
            current_lr = optimizer_CE1.param_groups[0]['lr']
            print(
                'Full HDCE:' f'SNR: {self.SNRdb} ' f'Epoch [{epoch}]/[{self.n_epochs}] learning rate: {current_lr:.4e}',
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            CE0.train()
            CE1.train()
            CE2.train()

            CE=[CE0,CE1,CE2]

            for it, (td00,td01,td02,td10,td11,td12,td20,td21,td22) in enumerate(train_loader):

                sutd = [[td00, td01, td02], [td10, td11, td12], [td20, td21, td22]]

                optimizer_CE0.zero_grad()
                optimizer_CE1.zero_grad()
                optimizer_CE2.zero_grad()

                total_loss=0
                total_loss_perf=0

                for sid in range(3):
                    for uid in range(3):
                        loss,loss_perf = self.get_loss(sutd[sid][uid],CE[sid],criterion,device)
                        loss = loss/9
                        loss_perf=loss_perf/9
                        total_loss+=loss
                        total_loss_perf+=loss_perf
                total_loss.backward()

                optimizer_CE0.step()
                optimizer_CE1.step()
                optimizer_CE2.step()


                if it % self.print_freq == 0:
                    print(
                        f'Epoch: [{epoch}/{self.n_epochs}][{it}/{len(train_loader)}]\t Loss {total_loss.item():.5f}\t Loss_perf {total_loss_perf.item():.5f}')

            CE0.eval()
            CE1.eval()
            CE2.eval()

            CE=[CE0,CE1,CE2]
            with torch.no_grad():
                Hhat_list = []
                Hlabel_list = []
                Hperfect_list = []

                for vd00,vd01,vd02,vd10,vd11,vd12,vd20,vd21,vd22 in val_loader:
                    suvd = [[vd00, vd01, vd02], [vd10, vd11, vd12], [vd20, vd21, vd22]]

                    for sid in range(3):
                        for uid in range(3):
                            Hhat,label_out,perfect_out=self.get_estimate(suvd[sid][uid],CE[sid],device)

                            Hhat_list.append(Hhat)
                            Hlabel_list.append(label_out)
                            Hperfect_list.append(perfect_out)

                Hhat = torch.cat(Hhat_list, dim=0)
                Hlabel = torch.cat(Hlabel_list, dim=0)
                Hperfect = torch.cat(Hperfect_list, dim=0)
                nmse = criterion(Hhat, Hlabel)
                nmse_perf = criterion(Hhat, Hperfect)


                if epoch==self.n_epochs-1:
                    torch.save({'ce': CE0.state_dict()},
                               os.path.join(f'./models/HDCE',
                                            f'CE0_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))
                    torch.save({'ce': CE1.state_dict()},
                               os.path.join(f'./models/HDCE',
                                            f'CE1_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))
                    torch.save({'ce': CE2.state_dict()},
                               os.path.join(f'./models/HDCE',
                                            f'CE2_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))
                    print('HDCE finally saved!')
                if nmse < best_nmse:
                    torch.save({'ce': CE0.state_dict()},
                               os.path.join(f'./models/HDCE',
                                            f'CE0_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))
                    torch.save({'ce': CE1.state_dict()},
                               os.path.join(f'./models/HDCE',
                                            f'CE1_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))
                    torch.save({'ce': CE2.state_dict()},
                               os.path.join(f'./models/HDCE',
                                            f'CE2_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))

                    best_nmse = nmse.item()
                    print('HDCE saved!')
                print(
                    f'Epoch [{epoch}]/[{self.n_epochs}] || NMSE {nmse.item():.5f}, NMSE_perf {nmse_perf.item():.5f}, best nmse: {best_nmse:.5f}')
            if epoch > 0:
                if epoch % self.lr_decay == 0:
                    optimizer_CE0.param_groups[0]['lr'] = optimizer_CE0.param_groups[0]['lr'] * 0.5
                    optimizer_CE1.param_groups[0]['lr'] = optimizer_CE1.param_groups[0]['lr'] * 0.5
                    optimizer_CE2.param_groups[0]['lr'] = optimizer_CE2.param_groups[0]['lr'] * 0.5
                if optimizer_CE0.param_groups[0]['lr'] < self.lr_threshold:
                    optimizer_CE0.param_groups[0]['lr'] = self.lr_threshold
                    optimizer_CE1.param_groups[0]['lr'] = self.lr_threshold
                    optimizer_CE2.param_groups[0]['lr'] = self.lr_threshold

if __name__ == '__main__':

    # Check if GPU is available and print the device info
    if torch.cuda.is_available():
        print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available. Using CPU.")

    train_nn = Model_train()
    train_nn.train_DCE_for_scenario(0)
    train_nn.train_DCE_for_scenario(1)
    train_nn.train_DCE_for_scenario(2)
    train_nn.train_DCE_for_DML()
    train_nn.train_Conv_Linear_of_HDCE()
    train_nn.train_SC_of_HDCE()
    train_nn.train_SDCE()