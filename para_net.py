import torch
import torch.nn as nn
import torch.optim as optim
import time
def compile_net(model,input_dim):
    model.eval()
    x = torch.randn(1,input_dim)
    traced_script_module = torch.jit.trace(model, x)
    return traced_script_module

def save_net(model,path,input_dim):
    model.eval()
    x = torch.randn(1,input_dim)
    traced_script_module = torch.jit.trace(model, x)
    torch.jit.save(traced_script_module,path)

def load_net(path,input_dim):
    model = torch.jit.load(path)
    model.eval()
    return model

class ParaNet(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(ParaNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)

def para_net_test(path,input_dim):
    net = load_net(path,input_dim)
    n_test = 100000
    time_start = time.time()
    for i in range(n_test):
        x = torch.randn(1,input_dim)
        y = net(x)
        # print(y)
    time_end = time.time()
    print(time_end - time_start)
    print(f"frequency: {n_test/(time_end - time_start)}")

def training_para_net(model, n_epochs, batch_size, transformation_train: torch.Tensor, wrench_train: torch.Tensor, transformation_test: torch.Tensor, wrench_test: torch.Tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)    
    transformation_train = transformation_train.to(device)
    wrench_train = wrench_train.to(device)
    transformation_test = transformation_test.to(device)
    wrench_test = wrench_test.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.9)
    trainset = torch.utils.data.TensorDataset(transformation_train, wrench_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    # testset = torch.utils.data.TensorDataset(transformation_test, wrench_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)
        # print(f"epoch {epoch + 1} loss: {loss.item()}")
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                out = model(transformation_test)
                val_loss = (out - wrench_test)**2
                val_loss = val_loss.mean()
                print(f"epoch {epoch + 1} val_loss: {val_loss.item()}")
                print(f"learning rate: {scheduler.get_last_lr()}")

    model.eval()
    model.cpu()
    save_net(model, "para_net.pt", 4)


if __name__ == "__main__":
    # model = ParaNet(4,6)
    # import numpy as np
    # import h5py 
    # file_path = '/home/ripl/workspace/ripl/ur5-tactile/data/collected_data_interactive_v2.h5'
    # wrenches = []
    # transformations = []
    # joints = []
    # with h5py.File(file_path, 'r') as f:
    #     demos = f['demos'] 
    #     for demo in demos:
    #         wrenches.append(np.array(demos[demo]['wrenches']))
    #         transformations.append(np.array(demos[demo]['transformations']))
    #         joints.append(np.array(demos[demo]['joint_positions']))
    # wrenches = np.concatenate(wrenches, axis=0)
    # transformations = np.concatenate(transformations, axis=0)[:,3:]
    # joints = np.concatenate(joints, axis=0)
    
    # n_samples = wrenches.shape[0]
    # n_train = int(n_samples * 1.0)
    # random_indices = np.random.permutation(n_samples)
    # wrench_train = torch.from_numpy(wrenches[random_indices[:n_train]]).float()
    # transformation_train = torch.from_numpy(transformations[random_indices[:n_train]]).float()
    # # wrench_test = torch.from_numpy(wrenches[random_indices[n_train:]]).float()
    # # transformation_test = torch.from_numpy(transformations[random_indices[n_train:]]).float()
    # wrench_test = wrench_train
    # transformation_test = transformation_train
    
    # training_para_net(model, 2000, 64, transformation_train, wrench_train, transformation_test, wrench_test)
    model = load_net("para_net.pt", 4)
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            print(f"{name}: dropout probability = {module.p}")
        
    
