import torch
import torch.nn as nn
import torch.optim as optim


class GravityCompensationModel(nn.Module):
    def __init__(self):
        super(GravityCompensationModel, self).__init__()
        
        # (1) Angles for world->base_link rotation
        # Initialize them to 0 or a guess (roll, pitch, yaw)
        self.angles_w_to_b = nn.Parameter(torch.zeros(3))
        # self.quat_w_to_b = nn.Parameter(torch.zeros(4))
        
        
        # (2) Payload mass
        self.mass = nn.Parameter(torch.tensor(1.0))
        
        # (3) Center of mass in base_link frame [cx, cy, cz]
        self.com = nn.Parameter(torch.zeros(3))
        
        # (4) Sensor offsets [fx0, fy0, fz0, tx0, ty0, tz0]
        self.offsets = nn.Parameter(torch.zeros(6))
        
        # initialize the parameters
        self.initialize_parameters()

    def initialize_parameters(self):
        # with small random noise
        self.angles_w_to_b.data.uniform_(-1, 1)
        self.mass.data.fill_(3.0)
        self.com.data.uniform_(-1, 1)
        self.offsets.data.uniform_(100, 200)
    
    def penalty_fn(self):
        # penalty for the parameters
        # 1. mass should be positive
        mass_penalty = torch.relu(-1.0 * self.mass.data)
        # 2. com should be within the range of the robot
        com_penalty = torch.relu(torch.abs(self.com.data) - 1.0)
        # 3. offsets should be small
        offsets_penalty_z = torch.relu(torch.abs(self.offsets.data[2]) - 0.1)
        offsets_penalty_x = torch.relu(torch.abs(self.offsets.data[0]) - 0.1)
        offsets_penalty_y = torch.relu(torch.abs(self.offsets.data[1]) - 0.1)
        return mass_penalty + com_penalty + offsets_penalty_z + offsets_penalty_x + offsets_penalty_y

    def forward(self, R_base_sensor, F_meas, T_meas):
        """
        R_base_sensor: shape [B, 3, 3], rotation from base_link -> sensor
        F_meas: shape [B, 3]
        T_meas: shape [B, 3]

        We'll:
          1) Convert angles_w_to_b -> R_w_to_b (3x3).
          2) Gravity in base_link frame: g_base = R_w_to_b * g_world.
          3) F_base = mass * g_base.
          4) tau_base = com x F_base.
          5) For each sample i, transform F_base, tau_base to sensor frame:
             F_sensor[i]   = R_base_sensor[i] * F_base
             tau_sensor[i] = R_base_sensor[i] * tau_base
          6) The "compensated" wrench = measured - (gravity_wrench + offset).
             We'll define an error that tries to push that to zero.
        """
        B = R_base_sensor.shape[0]
        # import pdb; pdb.set_trace()
        # (1) Angles -> rotation matrix from world base -> base_link
        R_w_to_b = self.euler_to_matrix(self.angles_w_to_b)
        # R_w_to_b = torch.eye(3)
        # R_w_to_b = self.quat_to_matrix(self.quat_w_to_b)
        # pdb.set_trace()
        # (2) Gravity in base_link frame
        g_world = torch.tensor([0.0, 0.0, -9.81])
        # (2) Gravity in base_link frame
        g_world_col = g_world.unsqueeze(1)  # Make g_world a column vector
        g_base = torch.matmul(R_w_to_b, g_world_col).squeeze(-1)  # shape [3]
        # pdb.set_trace()
        # (3) Force in base_link
        F_base = self.mass * g_base  # shape [3]
        # pdb.set_trace()
        # (4) Torque in base_link
        tau_base = torch.cross(self.com, F_base, dim=0)  # shape [3]
        # pdb.set_trace()
        # Expand to [B, 3] for batch multiplication
        F_base_expanded = F_base.unsqueeze(0).expand(B, -1)
        tau_base_expanded = tau_base.unsqueeze(0).expand(B, -1)

        # (5) Transform base_link->sensor for each sample
        # F_sensor[i] = R_base_sensor[i] * F_base
        F_sensor = torch.bmm(R_base_sensor, F_base_expanded.unsqueeze(-1)).squeeze(-1)
        tau_sensor = torch.bmm(R_base_sensor, tau_base_expanded.unsqueeze(-1)).squeeze(-1)

        # Combine force and torque into one [B, 6] gravity wrench
        gravity_wrench_sensor = torch.cat([F_sensor, tau_sensor], dim=1)  # [B, 6]

        # Expand offsets to [B, 6]
        offsets_expanded = self.offsets.unsqueeze(0).expand(B, -1)  # [B, 6]

        # Wrench measured: also shape [B, 6], so combine F_meas and T_meas
        W_meas = torch.cat([F_meas, T_meas], dim=1)  # [B, 6]

        # "Compensated" wrench = W_meas - gravity_wrench_sensor - offsets
        W_comp = W_meas - gravity_wrench_sensor - offsets_expanded

        return W_comp  # [B, 6], ideally near 0

    def quat_to_matrix(self, quat):
        """
        Convert a quaternion to a rotation matrix.
        """
        x, y, z, w = quat
        return torch.tensor([[w**2 + x**2 - y**2 - z**2, 2*(x*y - w*z), 2*(x*z + w*y)], 
                         [2*(x*y + w*z), w**2 - x**2 - y**2 + z**2, 2*(y*z - w*x)], 
                         [2*(x*z - w*y), 2*(y*z + w*x), w**2 - x**2 - y**2 - z**2]])
    
    @staticmethod
    def euler_to_matrix(angles):
        """
        Convert euler angles [roll, pitch, yaw] to a rotation matrix (3x3).
        We'll use Z-Y-X convention as an example: R = Rz * Ry * Rx.
        """
        roll, pitch, yaw = angles[0], angles[1], angles[2]

        Rx = torch.tensor([[1, 0, 0],
                           [0, torch.cos(roll), -torch.sin(roll)],
                           [0, torch.sin(roll),  torch.cos(roll)]])
        Ry = torch.tensor([[ torch.cos(pitch), 0, torch.sin(pitch)],
                           [0,                 1, 0               ],
                           [-torch.sin(pitch), 0, torch.cos(pitch)]])
        Rz = torch.tensor(
                            [[ torch.cos(yaw), -torch.sin(yaw), 0],
                           [ torch.sin(yaw),  torch.cos(yaw), 0],
                           [0,                0,               1]]
        )
       
        return Rz @ Ry @ Rx

if __name__ == "__main__":
    import numpy as np
    transformation_data = np.load("/home/ripl/workspace/ripl/ur5-tactile/wrench_calibration_data/transform_data_20250324_191103.npy", allow_pickle=True)
    wrench_data = np.load("/home/ripl/workspace/ripl/ur5-tactile/wrench_calibration_data/wrench_data_20250324_191103.npy", allow_pickle=True)
    
    # print(transformation_data)
    quaternion = transformation_data[:,3:]
    force = wrench_data[:, :3]
    torque = wrench_data[:, 3:]
    print(quaternion.shape)
    # convert quaternion to rotation matrix
    def quaternion_to_rotation_matrix(quaternion):
        """
        Convert a quaternion to a rotation matrix.
        """
        x, y, z, w = quaternion
        return np.array([[w**2 + x**2 - y**2 - z**2, 2*(x*y - w*z), 2*(x*z + w*y)], 
                         [2*(x*y + w*z), w**2 - x**2 - y**2 + z**2, 2*(y*z - w*x)], 
                         [2*(x*z - w*y), 2*(y*z + w*x), w**2 - x**2 - y**2 - z**2]])
    
    rotation_matrix = [quaternion_to_rotation_matrix(q) for q in quaternion]
    rotation_matrix = np.array(rotation_matrix)
    
    
    R_base_to_sensor = torch.tensor(rotation_matrix).float()
    F_measured = torch.tensor(force).float()
    T_measured = torch.tensor(torque).float()
    
    model = GravityCompensationModel()
    optimizer = optim.SGD(model.parameters(),lr=0.00001, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    # Suppose you have B samples
   


    def loss_fn(W_comp):
        # W_comp shape [B, 6], target is zero
        return (W_comp**2).mean()
    
    num_epochs = 500000

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        W_comp = model(R_base_to_sensor, F_measured, T_measured)
        
        # Loss: push W_comp to zero
        loss = loss_fn(W_comp).mean() + model.penalty_fn().mean()
        
        # Backprop + update
        loss.backward()
        optimizer.step()
        ## learning rate scheduler
        # scheduler.step(loss)
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
            # print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            

    print("\n=== Optimized Parameters ===")
    print("Angles (world->base) [roll, pitch, yaw]:", model.angles_w_to_b.detach().tolist())
    print("Mass:", model.mass.item())
    print("COM in base_link frame:", model.com.detach().tolist())
    print("Sensor offsets:", model.offsets.detach().tolist())