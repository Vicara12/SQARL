import torch
from random import randint
from sampler.hardwaresampler import HardwareSampler
from sampler.randomcircuit import RandomCircuit
from sqarl.sqarl import SQARL, ModelConfigs
from utils.customtypes import Hardware


def train_model_da(allocator, name: str):
  validation_hardware = Hardware(
    core_capacities=torch.tensor([4]*4),
    core_connectivity=(torch.ones(4,4) - torch.eye(4))
  )
  val_sampler = RandomCircuit(num_lq=16, num_slices=32)
  train_cfg = SQARL.TrainConfig(
    train_iters=30_000,
    batch_size=1,
    group_size=32,
    validate_each=25,
    validation_hardware=validation_hardware,
    validation_circuits=[val_sampler.sample() for _ in range(32)],
    store_path=f"trained/{name}",
    initial_noise=0.2,
    noise_decrease_factor=0.9995,
    min_noise=0.0,
    circ_sampler=RandomCircuit(num_lq=16, num_slices=lambda: randint(4,32)),
    lr=5e-5,
    inv_mov_penalization=0.0,
    mask_invalid=True,
    hardware_sampler=HardwareSampler(max_nqubits=16, range_ncores=[2,8]),
    dropout=0.0,
  )
  allocator.train(train_cfg)


def finetune_model_da(name: str):
  allocator = SQARL.load(f'trained/{name}', checkpoint=-1).set_mode(SQARL.Mode.Sequential)
  validation_hardware = Hardware(
    core_capacities=torch.tensor([4]*4),
    core_connectivity=(torch.ones(4,4) - torch.eye(4))
  )
  val_sampler = RandomCircuit(num_lq=16, num_slices=32)
  train_cfg = SQARL.TrainConfig(
    train_iters=30_000 - 4_200,
    batch_size=1,
    group_size=32,
    validate_each=25,
    validation_hardware=validation_hardware,
    validation_circuits=[val_sampler.sample() for _ in range(32)],
    store_path=f"trained/{name}_ft",
    initial_noise=0.002998814266877688,
    noise_decrease_factor=0.999,
    min_noise=0.0,
    circ_sampler=RandomCircuit(num_lq=16, num_slices=lambda: randint(8,32)),
    lr=2.5e-5,
    inv_mov_penalization=0.0,
    mask_invalid=True,
    hardware_sampler=HardwareSampler(max_nqubits=16, range_ncores=[2,8]),
    dropout=0.0,
  )
  allocator.train(train_cfg)



if __name__ == "__main__":
  allocator = SQARL(
    device='cuda',
    model_cfg=ModelConfigs(embed_size=64, num_heads=2, num_layers=2),
    mode=SQARL.Mode.Sequential,
  )
  train_model_da(allocator, name="da")