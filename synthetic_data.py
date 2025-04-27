import numpy as np
import math
import time
import os.path
from torch.utils.data import Dataset

from torch.utils.data import Dataset

class DatasetFolder(Dataset):
    def __init__(self, td):
        self.Yp, self.Hlabel, self.Hperf, self.Indicator = td

    def __len__(self):
        return len(self.Yp)

    def __getitem__(self, index):
        return [
            self.Yp[index],
            self.Hlabel[index],
            self.Hperf[index],
            self.Indicator[index]
        ]

class DatasetFolder_DML(Dataset):
    def __init__(self, *td_sets):
        # td_sets: (td00, td01, td02, td10, td11, td12, td20, td21, td22)
        self.td_sets = td_sets

    def __len__(self):
        return len(self.td_sets[0][0])

    def __getitem__(self, index):
        sample = []
        for td in self.td_sets:
            sample.append([
                td[0][index], 
                td[1][index], 
                td[2][index], 
                td[3][index]
            ])
        return sample

class RandomDataset(Dataset):
    def __init__(self, data_len, indicator, Pilot_num, SNRdb, seed_value=42):
        super().__init__()
        self.data_len = data_len
        self.indicator = indicator
        self.Pilot_num = Pilot_num
        self.SNRdb = SNRdb
        self.seed_value = seed_value

    def _generate_seed(self):
        return self.seed_value

    def __getitem__(self, index):
        np.random.seed(self._generate_seed())
        inin = self.indicator if self.indicator != -1 else np.random.randint(0, 3)
        hr = generate_hr(N1=8, N2=8, num_paths=3, index=inin)
        yy, hh, hperf = generate_data(hr, self.Pilot_num, self.SNRdb)
        return yy, hh, hperf, inin

    def __len__(self):
        return self.data_len



# Generate the steering vector for ULA
def ASV(N, theta):
    x = np.arange(N)[:, None]
    av = np.zeros((N, 1), dtype=np.complex64)
    for i in range(N):
        av[i] = (1 / np.sqrt(N)) * np.exp(-1j * np.pi * i * theta)
    return av

# Generate the steering vector for UPA
def generate_H_ASV(N1, N2, theta, phi):
    a1 = ASV(N1, np.cos(phi))
    a2 = ASV(N2, np.sin(phi) * np.cos(theta))
    result = np.zeros((N1 * N2, 1), dtype=np.complex64)
    for i in range(N1):
        for j in range(N2):
            result[i * N2 + j] = a1[i] * a2[j]
    return result

# Generate the two steering vectors for UPA
def generate_H_ASV1(Xt, Yt, Xr, Yr, theta_t, phi_t, theta_r, phi_r):
    at_x = ASV(Xt, np.cos(phi_t))
    at_y = ASV(Yt, np.sin(phi_t) * np.cos(theta_t))
    ar_x = ASV(Xr, np.cos(phi_r))
    ar_y = ASV(Yr, np.sin(phi_r) * np.cos(theta_r))
    ht = np.zeros((Xt * Yt, 1), dtype=np.complex64)
    for i in range(Xt):
        for j in range(Yt):
            ht[i * Yt + j] = at_x[i] * at_y[j]
    hr = np.zeros((Xr * Yr, 1), dtype=np.complex64)
    for i in range(Xr):
        for j in range(Yr):
            hr[i * Yr + j] = ar_x[i] * ar_y[j]
    H = np.zeros((Xr * Yr, Xt * Yt), dtype=np.complex64)
    for i in range(Xr * Yr):
        for j in range(Xt * Yt):
            H[i, j] = hr[i] * ht[j]
    return H

# Generate the channel from the RIS to the user
def generate_hr(N1, N2, num_paths, index):
    if index == 0:
        Phi = np.random.rand(num_paths) * (np.pi / 3) - (np.pi / 2)
    elif index == 1:
        Phi = np.random.rand(num_paths) * (np.pi / 3) + (np.pi / 3) - (np.pi / 2)
    elif index == 2:
        Phi = np.random.rand(num_paths) * (np.pi / 3) + (2 * np.pi / 3) - (np.pi / 2)
    else:
        raise ValueError("Index must be 0, 1, or 2.")
    Theta = np.random.rand(num_paths) * np.pi - (np.pi / 2)
    alpha = np.zeros(num_paths, dtype=np.complex64)
    for i in range(num_paths):
        alpha[i] = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    hr = np.zeros((N1 * N2, 1), dtype=np.complex64)
    for i in range(num_paths):
        hr += alpha[i] * generate_H_ASV(N1, N2, Theta[i], Phi[i])
    hr *= np.sqrt((N1 * N2) / num_paths)
    hr = hr.reshape(N1 * N2)
    return hr

#BS to RIS channel
def generate_G(M1, M2, N1, N2, num_paths):
    #num_paths number of multipath components
    # Define AoD and AoA (Azimuth and Elevation angles)
    Theta_t = np.random.rand(num_paths) * np.pi - np.pi / 2
    Phi_t = np.random.rand(num_paths) * np.pi - np.pi / 2
    Theta_r = np.random.rand(num_paths) * np.pi - np.pi / 2
    Phi_r = np.random.rand(num_paths) * np.pi - np.pi / 2

    # Define complex gain for each path
    alpha = (np.random.randn(num_paths) + 1j * np.random.randn(num_paths)) / np.sqrt(2)

    # Initialize the channel matrix G
    G = np.zeros((N1 * N2, M1 * M2), dtype=np.complex64)

    # Sum up the contributions of each path to G
    for i in range(num_paths):
        G += alpha[i] * generate_H_ASV1(M1, M2, N1, N2, Theta_t[i], Phi_t[i], Theta_r[i], Phi_r[i])

    # Scale the result
    G *= np.sqrt((N1 * N2 * M1 * M2) / num_paths)

    return G


def generate_data(hh, Pilot_num, SS):

    # hh (np.ndarray): Channel vector for the cascaded channel.
    G = np.load('training_data/G_64_16.npy')
    Psi = np.load(f'training_data/Psi_1024_{Pilot_num}.npy')
    # Compute the cascaded channel
    CascadedH = np.matmul(np.diag(hh), G)
    Cascadedh = np.reshape(CascadedH, [-1], order='F')
    # Calculate noise variance based on SNR
    sigma2 = 10 ** (-SS / 10)
    seed = 42
    np.random.seed(seed)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*Cascadedh.shape) + 1j * np.random.randn(*Cascadedh.shape))
    # Add noise to the cascaded channel to get the received signal
    hh = Cascadedh + noise
    # Compute the output signal (yy)
    yy = np.matmul(Psi, hh)
    #Return training data with the added noise
    return yy, hh, Cascadedh


def generate_datapair(Ns, Pilot_num, index, SNRdb):
    Yp, Hlabel, Hperfect, Indicator = [], [], [], []
    for i in range(Ns):
        inin = index if index != -1 else np.random.randint(0, 3)  # Randomize index if it's -1
        hr = generate_hr(N1=8, N2=8, num_paths=3, index=inin)      
        # Set SNR value based on input or randomize it
        # Generate data (yy, hh, hperf)
        yy, hh, hperf = generate_data(hr, Pilot_num, SNRdb)
        # Append the results to the respective lists
        Yp.append(yy)
        Hlabel.append(hh)
        Hperfect.append(hperf)
        Indicator.append(inin)
    # Stack the lists into numpy arrays
    Yp = np.stack(Yp, axis=0)
    Hlabel = np.stack(Hlabel, axis=0)
    Hperfect = np.stack(Hperfect, axis=0)
    Indicator = np.stack(Indicator, axis=0)
    return Yp, Hlabel, Hperfect, Indicator


def generate_MMSE_estimate(Hhat_LS, sigma2):
    Sample_num = len(Hhat_LS)
    # Initialize the covariance matrix Rh
    Rh = np.zeros((1024, 1024), dtype=np.complex64)
    # Compute the covariance matrix Rh from the LS estimates
    for s in range(Sample_num):
        temph = Hhat_LS[s].reshape([-1, 1])
        Rh += np.matmul(temph, temph.T.conj())  # Accumulate outer products
    # Normalize Rh by the number of samples
    Rh /= Sample_num
    # Compute the MMSE estimate using the formula
    Rh_inv = np.linalg.inv(Rh + (sigma2 * np.eye(len(Rh))))  # Inverse of Rh + noise term
    Hhat_MMSE = np.matmul(np.matmul(Rh, Rh_inv), Hhat_LS.T).T  # MMSE estimation
    return Hhat_MMSE

def generate_data_files(M1, M2, N1, N2, data_len, SNRdb):
    data_dir = 'training_data'
    os.makedirs(data_dir, exist_ok=True)

    # Generate and save G matrix
    G = generate_G(M1=M1, M2=M2, N1=N1, N2=N2, num_paths=3)
    G_file = os.path.join(data_dir, f'G_{N1*N2}_{M1*M2}.npy')
    np.save(G_file, G)
    print(f'{G_file} has been saved!')

    # Generate and save Psi matrix
    Pilot_num = 128
    Psi = np.sqrt(1 / Pilot_num) * (2 * (np.random.rand(Pilot_num, M1*M2*N1*N2) > 0.5) - 1)
    Psi_file = os.path.join(data_dir, f'Psi_{M1*M2*N1*N2}_{Pilot_num}.npy')
    np.save(Psi_file, Psi)
    print(f'{Psi_file} has been saved!')

    # Generate and save data for each scenario and user
    for sid in range(3):
        for uid in range(3):
            print(f'Generating data for scenario {sid} when Pilot_num={Pilot_num} and User_id={uid}!')
            Yp, Hlabel, Hperf, Indicator = generate_datapair(Ns=data_len, Pilot_num=Pilot_num, index=sid, SNRdb=SNRdb)

            for name, data in zip(['Yp', 'Hlabel', 'Hperf'], [Yp, Hlabel, Hperf]):
                file_path = os.path.join(
                    data_dir, f'{name}{sid}_{Pilot_num}_1024_{SNRdb}dB_{uid}_datalen_{data_len}.npy'
                )
                np.save(file_path, data)
            print(f'Data for scenario {sid} when Pilot_num={Pilot_num} and User_id={uid} has been saved!')


if __name__ == '__main__':
    M1, M2 = 4, 4 # 16 antennas at BS
    N1, N2 = 8, 8 # 64 antennas at RIS
    data_len = 10000
    SNRdb = 10

    generate_data_files(M1, M2, N1, N2, data_len, SNRdb)

