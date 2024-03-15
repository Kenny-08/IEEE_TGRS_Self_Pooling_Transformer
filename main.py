from loop_train_test import loop_train_test

def resolve_hp(hp: dict):
    return hp.get('run_times'), hp.get('num_PC'), hp.get('train_num'), \
           hp.get('patch_size'), hp.get('batch_size'), hp.get('lr'), \
           hp.get('epoch'), hp.get('autoencoder_path')


def IP_experiment(hp: dict):
    run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, autoencoder_path = resolve_hp(hp)
    loop_train_test('IP', run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, autoencoder_path)


def PU_experiment(hp: dict):
    run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, autoencoder_path = resolve_hp(hp)
    loop_train_test('PU', run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, autoencoder_path)


def Salinas_experiment(hp: dict):
    run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, autoencoder_path = resolve_hp(hp)
    loop_train_test('Salinas', run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, autoencoder_path)


def HU_experiment(hp: dict):
    run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, autoencoder_path = resolve_hp(hp)
    loop_train_test('Houston', run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, autoencoder_path)

def HongHu_experiment(hp: dict):
    run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, autoencoder_path = resolve_hp(hp)
    loop_train_test('HongHu', run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, autoencoder_path)

def HanChuan_experiment(hp: dict):
    run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, autoencoder_path = resolve_hp(hp)
    loop_train_test('HanChuan', run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, autoencoder_path)

def LongKou_experiment(hp: dict):
    run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, autoencoder_path = resolve_hp(hp)
    loop_train_test('LongKou', run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, autoencoder_path)


if __name__ == '__main__':
    import os
    hyperparameter_pu = {
        'run_times': 1,
        'num_PC': 32,
        'train_num': 5,
        'patch_size': 11,
        'batch_size': 45,
        'lr': 1e-3,
        'epoch': 145,
        'autoencoder_path': './save/autoencoder/PU/PU_32.pt',
    }
    # PU_experiment(hp=hyperparameter_pu)

    hyperparameter_ip = {
        'run_times': 1,
        'num_PC': 40,
        'train_num': 5,
        'patch_size': 13,
        'batch_size': 80,
        'lr': 1e-3,
        'epoch': 155,
        'autoencoder_path': './save/autoencoder/IP/IP_40.pt',
    }
    # IP_experiment(hp=hyperparameter_ip)

    hyperparameter_salinas = {
        'run_times': 1,
        'num_PC': 40,
        'train_num': 5,
        'patch_size': 13,
        'batch_size': 80,
        'lr': 1e-3,
        'epoch': 165,
        'autoencoder_path': './save/autoencoder/Salinas/Salinas_40.pt',
    }
    # Salinas_experiment(hp=hyperparameter_salinas)

    hyperparameter_hu = {
        'run_times': 1,
        'num_PC': 32,
        'train_num': 5,
        'patch_size': 11,
        'batch_size': 75,
        'lr': 1e-3,
        'epoch': 215,
        'autoencoder_path': './save/autoencoder/Houston/HU_32.pt',
    }
    # HU_experiment(hp=hyperparameter_hu)

    hyperparameter_longkou = {
        'run_times': 1,
        'num_PC': 32,
        'train_num': 3,
        'patch_size': 11,
        'batch_size': 75,
        'lr': 1e-3,
        'epoch': 200,
        'autoencoder_path': '/content/IEEE_TGRS_Self_Pooling_Transformer/save/autoencoder/epoch-99train_loss-0.0031834827876070634.pt',
    }
    LongKou_experiment(hp=hyperparameter_longkou)

    hyperparameter_hanchuan = {
        'run_times': 1,
        'num_PC': 32,
        'train_num': 5,
        'patch_size': 11,
        'batch_size': 75,
        'lr': 1e-3,
        'epoch': 200,
        'autoencoder_path': './save/autoencoder/HanChuan/xyz.pt',
    }
    # HanChuan_experiment(hp=hyperparameter_hanchuan)

    hyperparameter_honghu = {
        'run_times': 1,
        'num_PC': 32,
        'train_num': 5,
        'patch_size': 11,
        'batch_size': 75,
        'lr': 1e-3,
        'epoch': 200,
        'autoencoder_path': './save/autoencoder/HongHu/xyz.pt',
    }
    # HongHu_experiment(hp=hyperparameter_honghu)
