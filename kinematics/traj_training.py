import os
import pickle
import time

os.environ['KERAS_BACKEND'] = 'torch'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from keras.api.metrics import mean_squared_error
from keras.api.models import Model, load_model
from torch import nn, Tensor
from torch.optim import Optimizer, Adam, SGD, RMSprop
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# seed = 42
# np.random.seed(seed)
# torch.manual_seed(seed)
# keras.utils.set_random_seed(seed)


class TrajModel(nn.Module):
    def __init__(
            self,
            input_dim: int,
            n_timesteps: int,
            n_gru: int,
            d_gru: int,
            out_layers: dict[str, int],
            device: str
    ) -> None:
        super(TrajModel, self).__init__()

        self.input_dim = input_dim
        self.n_timesteps = n_timesteps
        self.n_gru = n_gru
        self.d_gru = d_gru

        self.device = device

        self.grus = [
            nn.GRUCell(
                input_size=self.input_dim,
                hidden_size=self.d_gru,
                device=self.device
            )
        ] + [
            nn.GRUCell(
                input_size=self.d_gru,
                hidden_size=self.d_gru,
                device=self.device
            )
            for _ in range(self.n_gru - 1)
        ]
        self.lin_common = nn.Linear(self.d_gru, self.d_gru, device=self.device)
        self.out_heads: dict[str, tuple[nn.Module, int]] = {
            out_name: (
                nn.Sequential(
                    nn.Linear(self.d_gru, 10, device=self.device),
                    nn.Tanh(),
                    nn.Linear(10, shape, device=self.device)
                ),
                shape
            )
            for out_name, shape in out_layers.items()
        }

    def forward(self, X: Tensor) -> dict[str, Tensor]:
        assert X.dim() == 2
        assert X.shape[1] == self.input_dim

        batch_size = X.shape[0]

        h_grus = [
            torch.randn(batch_size, self.d_gru).to(self.device)
            for _ in range(self.n_gru)
        ]
        outputs: dict[str, Tensor] = {
            out_name: torch.zeros(batch_size, self.n_timesteps, dim).to(self.device)
            for out_name, (_, dim) in self.out_heads.items()
        }

        for t in range(self.n_timesteps):
            h = X
            for gru_id, gru in enumerate(self.grus):
                h = gru(h, h_grus[gru_id])
                h_grus[gru_id] = h

            h = self.lin_common(h)
            h = F.tanh(h)

            for out_name, (head, out_dim) in self.out_heads.items():
                outputs[out_name][:, t] = head(h)

        return outputs


class KinematicsExperiment:
    def __init__(
            self,
            fwd_model_path: str,
            inv_model_path: str,
            data_path: str,
            pretrain_size: float = 0.5,
            n_timesteps: int = 25,
            traj_model_path: str = None
    ) -> None:
        assert 0.0 <= pretrain_size <= 1.0
        assert 0 < n_timesteps

        self.pretrain_size = pretrain_size
        self.n_timesteps = n_timesteps

        self.X_pre, self.X_train, self.y_pre = self._process_data(data_path)

        self.d_gru: int = 20
        self.n_gru: int = 1

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.traj_model: TrajModel | None = None
        self.fwd_model: Model = load_model(fwd_model_path)
        self.inv_model: Model = load_model(inv_model_path)
        self.fwd_model.trainable = False
        self.inv_model.trainable = False

        if traj_model_path:
            traj_model: TrajModel = torch.load(traj_model_path, weights_only=False)

            assert traj_model.grus[0].input_size == self.X_train.shape[1]
            # assert all(
            #     self.y_pre.get(out_name) is not None
            #     and self.y_pre[out_name].shape[1:] == traj_model.output[idx].shape[1:]
            #     for idx, (out_name, *_) in enumerate(traj_model.out_heads)
            # )

            self.traj_model = traj_model
            self.d_gru = traj_model.d_gru
            self.n_gru = traj_model.n_gru

    def _process_data(self, data_path: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        with h5py.File(data_path, 'r') as hf:
            endpoints = hf['kuka_kinematics_endpoints'][()]
            trajS = hf['kuka_kinematics_trajS'][()]

        assert trajS.ndim == 3
        assert trajS.shape[1] >= self.n_timesteps
        assert trajS.shape[2] == 10

        pretrain_idx = int(trajS.shape[0] * self.pretrain_size)

        endpoints_pretrain, endpoints_train = np.vsplit(endpoints, [pretrain_idx])
        trajS_pretrain, _ = np.vsplit(trajS[:, :self.n_timesteps], [pretrain_idx])

        y_traj_joints, y_traj_eff_xyz = np.split(trajS_pretrain, [7], axis=2)

        y_pre = {
            'joints_out': y_traj_joints,
            'eff_xyz_out': y_traj_eff_xyz
        }

        return endpoints_pretrain, endpoints_train, y_pre

    def _construct_model(self) -> nn.Module:
        return TrajModel(
            input_dim=self.X_pre.shape[1],
            n_timesteps=self.n_timesteps,
            n_gru=self.n_gru,
            d_gru=self.d_gru,
            out_layers={
                out_name: data.shape[2]
                for out_name, data in self.y_pre.items()
                if out_name == 'eff_xyz_out'
            },
            device=self.device
        )

    def _log_name(
            self,
            n_epochs: int,
            optimizer: Optimizer,
            pre: bool = False,
            gamma: float | str = None
    ) -> str:
        momentum = (
            f',momentum={str(optimizer.param_groups[0]["momentum"])}'
            if isinstance(optimizer, SGD)
            else ''
        )

        return ''.join([
            'traj_',
            f'gru{self.n_gru}-d={self.d_gru}-head=10_',
            f'traj={self.n_timesteps}_',
            f'custom-{"pre" if pre else "full-no-pre"}_',
            f'{n_epochs}ep_',
            f'{optimizer.__class__.__name__}(lr={str(optimizer.param_groups[0]["lr"])}{momentum})',
            f'_gamma={gamma}' if gamma is not None else '',
            '_closest-si0-sf0-err_no-intermediate-err_eff-only'
        ])

    def pretrain(
            self,
            log_dir: str,
            optimizer: str = 'adam',
            lr: float = 1e-3,
            batch_size: int = 256,
            n_epochs: int = 1,
            save_dir: str = None
    ) -> None:
        assert self.traj_model is None
        self.traj_model = self._construct_model()

        train_idx = int(self.X_pre.shape[0] * 0.8)

        train = TensorDataset(
            torch.from_numpy(self.X_pre[:train_idx]).float().to(self.device),
            # torch.from_numpy(self.y_pre['joints_out'][:train_idx]).float().to(self.device),
            torch.from_numpy(self.y_pre['eff_xyz_out'][:train_idx]).float().to(self.device),
        )
        val = TensorDataset(
            torch.from_numpy(self.X_pre[train_idx:]).float().to(self.device),
            # torch.from_numpy(self.y_pre['joints_out'][train_idx:]).float().to(self.device),
            torch.from_numpy(self.y_pre['eff_xyz_out'][train_idx:]).float().to(self.device)
        )

        train_loader = DataLoader(
            dataset=train,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            dataset=val,
            batch_size=batch_size,
            shuffle=True
        )

        if optimizer == 'adam':
            optimizer = Adam(
                params=self.traj_model.parameters(),
                lr=lr
            )
        elif optimizer == 'sgd':
            optimizer = SGD(
                params=self.traj_model.parameters(),
                lr=lr
            )
        else:
            raise NotImplementedError

        mse = nn.MSELoss()
        mae = nn.L1Loss()

        writer = SummaryWriter(
            log_dir=os.path.join(
                log_dir,
                f'{self._log_name(n_epochs, optimizer, pre=True)}'
            )
        )

        for epoch in range(n_epochs):
            ep_metrics = {
                'loss': [],
                # 'mse_joints': [],
                'mse_eff': [],
                'mae': [],
                # 'mae_joints': [],
                'mae_eff': []
            }

            self.traj_model.train()

            # for batch_idx, (X, y_joints, y_eff) in enumerate(train_loader):
            for batch_idx, (X, y_eff) in enumerate(train_loader):
                y_pred = self.traj_model(X)

                # mse_joints = mse(y_pred['joints_out'], y_joints)
                mse_eff = mse(y_pred['eff_xyz_out'], y_eff)
                # loss = (mse_joints + mse_eff) / 2
                loss = mse_eff

                # mae_joints = mae(y_pred['joints_out'], y_joints)
                mae_eff = mae(y_pred['eff_xyz_out'], y_eff)
                # mae_all = (mae_joints + mae_eff) / 2
                mae_all = mae_eff

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ep_metrics['loss'].append(loss.item())
                # ep_metrics['mse_joints'].append(mse_joints.item())
                ep_metrics['mse_eff'].append(mse_eff.item())
                ep_metrics['mae'].append(mae_all.item())
                # ep_metrics['mae_joints'].append(mae_joints.item())
                ep_metrics['mae_eff'].append(mae_eff.item())

                step = epoch * len(train_loader) + batch_idx

                writer.add_scalar('loss-step/train', loss, step)
                # writer.add_scalar('mse_joints-step/train', mse_joints, step)
                writer.add_scalar('mse_eff-step/train', mse_eff, step)
                writer.add_scalar('mae-step/train', mae_all, step)
                # writer.add_scalar('mae_joints-step/train', mae_joints, step)
                writer.add_scalar('mae_eff-step/train', mae_eff, step)

                if batch_idx % 10 == 0:
                    print(
                        f'Train Epoch: {epoch} '
                        f'[{batch_idx * len(X)}/{len(train_loader.dataset)} ({batch_idx / len(train_loader):.1%})]\t'
                        # f'MSE: {loss.item():.6f} (joints: {mse_joints.item():.6f}, eff: {mse_eff.item():.6f})\t'
                        # f'MAE: {mae_all.item():.6f} (joints: {mae_joints.item():.6f}, eff: {mae_eff.item():.6f})'
                        f'MSE: {loss.item():.6f}\t'
                        f'MAE: {mae_all.item():.6f}'
                    )

            writer.add_scalar('loss/train', np.mean(ep_metrics['loss']), epoch)
            # writer.add_scalar('mse_joints/train', np.mean(ep_metrics['mse_joints']), epoch)
            writer.add_scalar('mse_eff/train', np.mean(ep_metrics['mse_eff']), epoch)
            writer.add_scalar('mae/train', np.mean(ep_metrics['mae']), epoch)
            # writer.add_scalar('mae_joints/train', np.mean(ep_metrics['mae_joints']), epoch)
            writer.add_scalar('mae_eff/train', np.mean(ep_metrics['mae_eff']), epoch)

            ep_metrics = {
                'loss': [],
                # 'mse_joints': [],
                'mse_eff': [],
                'mae': [],
                # 'mae_joints': [],
                'mae_eff': []
            }

            self.traj_model.eval()

            with torch.no_grad():
                # for X, y_joints, y_eff in val_loader:
                for X, y_eff in val_loader:
                    y_pred = self.traj_model(X)

                    # mse_joints = mse(y_pred['joints_out'], y_joints)
                    mse_eff = mse(y_pred['eff_xyz_out'], y_eff)
                    # loss = (mse_joints + mse_eff) / 2
                    loss = mse_eff

                    # mae_joints = mae(y_pred['joints_out'], y_joints)
                    mae_eff = mae(y_pred['eff_xyz_out'], y_eff)
                    # mae_all = (mae_joints + mae_eff) / 2
                    mae_all = mae_eff

                    ep_metrics['loss'].append(loss.item())
                    # ep_metrics['mse_joints'].append(mse_joints.item())
                    ep_metrics['mse_eff'].append(mse_eff.item())
                    ep_metrics['mae'].append(mae_all.item())
                    # ep_metrics['mae_joints'].append(mae_joints.item())
                    ep_metrics['mae_eff'].append(mae_eff.item())

            writer.add_scalar('loss/val', np.mean(ep_metrics['loss']), epoch)
            # writer.add_scalar('mse_joints/val', np.mean(ep_metrics['mse_joints']), epoch)
            writer.add_scalar('mse_eff/val', np.mean(ep_metrics['mse_eff']), epoch)
            writer.add_scalar('mae/val', np.mean(ep_metrics['mae']), epoch)
            # writer.add_scalar('mae_joints/val', np.mean(ep_metrics['mae_joints']), epoch)
            writer.add_scalar('mae_eff/val', np.mean(ep_metrics['mae_eff']), epoch)

        writer.close()

        if save_dir:
            torch.save(
                self.traj_model,
                os.path.join(
                    save_dir,
                    f'{self._log_name(n_epochs, optimizer, pre=True)}.pth'
                )
            )

    def train(
            self,
            optimizer: str = 'adam',
            lr=1e-3,
            batch_size: int = 256,
            n_epochs: int = 100,
            gamma: float | str = 1.0,
            log_dir: str = None,
            save_dir: str = None,
            save_weights: bool = False,
            verbose: bool = True
    ) -> None:
        assert isinstance(gamma, float) or gamma == 'csch'

        pretrained = True

        if self.traj_model is None:
            self.traj_model = self._construct_model()
            pretrained = False

        train_dataset = TensorDataset(torch.from_numpy(self.X_train).float().to(self.device))
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        if optimizer == 'adam':
            optimizer = Adam(
                params=self.traj_model.parameters(),
                lr=lr
            )
        elif optimizer == 'sgd':
            optimizer = SGD(
                params=self.traj_model.parameters(),
                lr=lr
            )
        elif optimizer == 'rmsprop':
            optimizer = RMSprop(
                params=self.traj_model.parameters(),
                lr=lr
            )
        else:
            raise NotImplementedError

        mse = nn.MSELoss()
        mae = nn.L1Loss()

        ep_losses = []
        ep_metrics = []
        ep_data = []

        log_dir = os.path.join(
            log_dir,
            ''.join([
                f'{self._log_name(n_epochs, optimizer, pre=pretrained, gamma=gamma)}',
                '' if pretrained else '_no-pre'
            ])
        )

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(os.path.join(log_dir, 'weights')):
            os.makedirs(os.path.join(log_dir, 'weights'))

        log_file_id = 0

        self.traj_model.train()

        for epoch in tqdm(range(n_epochs), desc='Epoch'):
            step_loss = []
            step_metric = []
            step_data = []
            step_dt = []

            for step, endpoint_batch in enumerate(tqdm(train_loader, desc='Step')):
                # endpoint_batch: Tensor = endpoint_batch.to(device=self.device)
                endpoint_batch: Tensor = endpoint_batch[0]

                # Forward pass
                t0 = time.time()

                y_pred: dict[str, Tensor] = self.traj_model(endpoint_batch)
                y_pred: list[Tensor] = [
                    # y_pred['joints_out'],
                    y_pred['eff_xyz_out']
                ]

                step_dt.append(time.time() - t0)

                s_init, s_final = torch.hsplit(endpoint_batch, 2)

                with torch.no_grad():
                    y_rectified = self._generate_y(y_pred, s_init, s_final)

                y_pred: Tensor = torch.concatenate(y_pred, dim=2)
                losses = self._L_TM2(y_pred, y_rectified, s_init, s_final, gamma=gamma)
                loss = torch.mean(losses)

                sf_dists = self._L_FM(s_final, y_rectified[:, -1]).cpu().numpy()
                sf_dist = np.mean(sf_dists)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_loss.append(float(loss))
                step_metric.append(sf_dist)
                step_data.append({
                    's_init': s_init.cpu().numpy(),
                    's_final': s_final.cpu().numpy(),
                    'y_pred': y_pred.detach().cpu().numpy(),
                    'y_rectified': y_rectified.cpu().numpy(),
                    'L_TM2': losses.detach().cpu().numpy(),
                    'sf_dists': sf_dists
                })

            ep_loss = np.mean(step_loss)
            ep_metric = np.mean(step_metric)

            ep_losses.append(ep_loss)
            ep_metrics.append(ep_metric)
            ep_data.append(step_data)

            if verbose:
                print(f'Epoch {epoch} done \t\t L_TM2: {ep_loss:.4f}, closeness to final state: {ep_metric:.4f}')
                print(f'TM inference on {self.device} took {np.mean(step_dt)} +- {np.std(step_dt)}')

            if save_weights:
                torch.save(
                    self.traj_model.state_dict(),
                    os.path.join(log_dir, 'weights', f'ep_{epoch}.weights.pth')
                )

            if epoch > 0 and epoch % (n_epochs // 10) == 0:
                with open(os.path.join(log_dir, f'training_{log_file_id}.pkl'), 'wb') as f:
                    pickle.dump((ep_losses, ep_metrics, ep_data), f)

                log_file_id += 1

                ep_losses = []
                ep_metrics = []
                ep_data = []

        torch.save(
            self.traj_model,
            os.path.join(
                save_dir,
                f'{self._log_name(n_epochs, optimizer, pre=False, gamma=gamma)}{"" if pretrained else "_no-pre"}.pth'
            )
        )

        if ep_data:
            with open(os.path.join(log_dir, f'training_{log_file_id}.pkl'), 'wb') as f:
                pickle.dump((ep_losses, ep_metrics, ep_data), f)

    def _generate_y(self, y_pred: list[Tensor], s_init: Tensor, s_final: Tensor) -> Tensor:
        n_samples, T, _ = y_pred[0].shape
        # joints_out_idx = 0
        joints_out_idx = -1
        y_rectified = []

        for i in range(n_samples):
            # im_inputs = []
            si = s_init[i]
            sf = s_final[i]

            # Joint subvector removal from the endpoint states
            partial_si = si[7:]
            partial_sf = sf[7:]

            s_rect = [si]
            s_pred = [partial_si]

            # Generated trajectory pre-processing
            for t in range(T):
                # Line 4
                comps = [
                    comp[i][t]
                    for comp in y_pred
                ]
                # full_s.append(torch.concatenate(comps))

                # Line 5
                s_pred.append(
                    torch.concatenate([
                        comp
                        for j, comp in enumerate(comps)
                        if j != joints_out_idx
                    ])
                )

            # full_s = torch.vstack(full_s + [sf])
            s_pred = torch.vstack(s_pred + [partial_sf])

            for t in range(T + 1):
                s0 = s_rect[t]
                s1 = s_pred[t + 1]

                # im_inputs.append(torch.concatenate((s0, s1)))

                action: Tensor = torch.from_numpy(
                    self.inv_model.predict(
                        torch.atleast_2d(torch.concatenate((s0, s1))),
                        verbose=False
                    )[0]
                ).to(device=self.device)

                # fm_inputs = torch.hstack([full_s[:-1], actions])

                s1_rect = torch.from_numpy(
                    np.concatenate(
                        self.fwd_model.predict(
                            torch.atleast_2d(torch.concatenate((s0, action))),
                            verbose=False
                        ),
                        axis=1
                    )[0]
                ).to(device=self.device)
                s_rect.append(s1_rect)

            y_rectified.append(torch.vstack(s_rect[1:]))

        return torch.stack(y_rectified)

    @staticmethod
    def _L_FM(y_pred: Tensor, y_true: Tensor) -> Tensor:
        partition_ids = [7]
        # y_true = torch.hsplit(y_true, partition_ids)
        # y_pred = torch.hsplit(y_pred, partition_ids)

        eff_true = (
            torch.hsplit(y_true, partition_ids)[1]
            if y_true.shape[1] == 10
            else y_true
        )

        eff_pred = (
            torch.hsplit(y_pred, partition_ids)[1]
            if y_pred.shape[1] == 10
            else y_pred
        )

        # mses = torch.vstack([
        #     mean_squared_error(y_true[i], y_pred[i])
        #     for i in range(len(y_true))
        # ])
        # l_fm = torch.mean(mses, dim=0)

        l_fm = mean_squared_error(eff_true, eff_pred)

        return l_fm

    def _L_TM2(
            self,
            y_pred: Tensor,
            y_rectified: Tensor,
            si_true: Tensor,
            sf_true: Tensor,
            gamma: float | str = 1.0
    ) -> Tensor:
        steps_pred = y_pred.shape[1]
        # boost = (
        #     (lambda t: gamma ** t)
        #     if isinstance(gamma, float)
        #     else (lambda t: 1 / np.sinh(-t + 26.1) + 1)
        # )

        net_err = torch.sum(
            torch.vstack([
                # boost(i) * self._L_FM(y_pred[:, i], y_rectified[:, i])
                self._L_FM(y_pred[:, i], y_rectified[:, i])
                for i in range(steps_pred)
            ]),
            dim=0
        )

        si_dists = self._L_FM(y_pred[:, 0], si_true)
        sf_dists = self._L_FM(y_pred[:, -1], sf_true)
        # sf_dists = torch.vstack([
        #     self._L_FM(y_pred[:, i], sf_true)
        #     for i in range(steps_pred)
        # ]).min(dim=0).values

        # mean_err = net_err / steps_pred + boost(steps_pred) * self._L_FM(sf_true, y_rectified[:, -1])
        # mean_err = net_err / steps_pred + si_dists + sf_dists
        mean_err = si_dists + sf_dists
        # mean_err = (net_err + boost(steps_pred) * self._L_FM(sf_true, y_rectified[:, -1])) / (steps_pred + 1)

        return mean_err


if __name__ == '__main__':
    experiment = KinematicsExperiment(
        fwd_model_path='../models/kinematics/kuka_fwd_60ep.keras',
        inv_model_path='../models/kinematics/kuka_inv_sep_100ep.keras',
        # traj_model_path='../models/kinematics/traj_gru1-d=100_traj=25_custom-pre_1ep_Adam(lr=0.001).pth',
        # traj_model_path='../models/kinematics/traj_gru1-d=100_traj=25_custom-pre_1ep_Adam(lr=0.001)_last-err-isolated_eff-only.pth',
        data_path='../data/kinematics/kuka_expert_traj_12k_n10_v2_f10_edge-pad.h5',
        pretrain_size=0.0,
        n_timesteps=10
    )

    # experiment.pretrain(
    #     log_dir='../logs/kinematics',
    #     optimizer='adam',
    #     lr=0.001,
    #     batch_size=256,
    #     n_epochs=1,
    #     save_dir='../models/kinematics'
    # )

    # optimizer = Adam(learning_rate=1e-3)
    # optimizer = RMSprop(learning_rate=1e-3)
    # optimizer = SGD(learning_rate=1e-3)
    # optimizer = SGD(learning_rate=5e-2)
    # optimizer = SGD(learning_rate=1e-2)
    # optimizer = SGD(learning_rate=1e-4)

    experiment.train(
        optimizer='rmsprop',
        lr=0.001,
        batch_size=1,
        n_epochs=150,
        gamma=1.0,
        log_dir='../logs/kinematics/full-logs',
        save_dir='../models/kinematics',
        save_weights=True,
        verbose=True
    )
