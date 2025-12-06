import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -------------------------- 1. 物理参数与配置（无修改） --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L = 0.02
T0 = 293.15
T_env = 293.15
T_heat = 353.15
h = 50.0

rho = 1800.0
cp = 1200.0
k0 = 0.2
k1 = 0.8
Q = 40000.0
A = 1e5  # 降低量级
Ea = 80000.0
m = 0.5
n = 1.5
R = 8.314

num_boundary = 200
num_initial = 200
num_collocation = 5000
num_experiment = 100
epochs = 8000
lr = 1e-4
weight_data = 1.0
weight_pde = 0.1
clip_norm = 1.0

x_scale = L
t_scale = 3600
T_mean = 300.0
T_scale = 100.0


# -------------------------- 2. 生成训练数据（无修改） --------------------------
def generate_data():
    x_init = torch.rand(num_initial, 1) * L
    t_init = torch.zeros_like(x_init)
    T_init = torch.full_like(x_init, T0)
    alpha_init = torch.zeros_like(x_init)

    t_left = torch.rand(num_boundary // 2, 1) * 3600
    x_left = torch.zeros_like(t_left)
    T_left = torch.full_like(t_left, T_heat)
    alpha_left = torch.rand_like(t_left)

    t_right = torch.rand(num_boundary // 2, 1) * 3600
    x_right = torch.full_like(t_right, L)
    T_right = torch.rand_like(t_right) * (T_heat - T0) + T0
    alpha_right = torch.rand_like(t_right)

    x_col = torch.rand(num_collocation, 1) * L
    t_col = torch.rand(num_collocation, 1) * 3600

    x_exp = torch.rand(num_experiment, 1) * L
    t_exp = torch.rand(num_experiment, 1) * 3600
    alpha_exp = 1 - torch.exp(- (A * t_exp) * torch.exp(-Ea / (R * (T0 + (T_heat - T0) * t_exp / 3600))))
    alpha_exp = alpha_exp.clamp(1e-6, 1 - 1e-6)
    T_exp = T0 + (T_heat - T0) * (1 - x_exp / L) + (Q * alpha_exp * rho) / (cp * rho)
    T_exp = T_exp.clamp(200, 400)

    return (
        x_init.to(device), t_init.to(device), T_init.to(device), alpha_init.to(device),
        x_left.to(device), t_left.to(device), T_left.to(device),
        x_right.to(device), t_right.to(device),
        x_col.to(device), t_col.to(device),
        x_exp.to(device), t_exp.to(device), T_exp.to(device), alpha_exp.to(device)
    )


data = generate_data()
x_init, t_init, T_init, alpha_init, x_left, t_left, T_left, x_right, t_right, x_col, t_col, x_exp, t_exp, T_exp, alpha_exp = data


# -------------------------- 3. 构建PINN模型（无修改） --------------------------
class PropellantPINN(nn.Module):
    def __init__(self, hidden_dim=100, num_layers=4):
        super(PropellantPINN, self).__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        self.feature_extractor = nn.Sequential(*layers)
        self.T_output = nn.Linear(hidden_dim, 1)
        self.alpha_output = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x, t):
        x_norm = x / x_scale
        t_norm = t / t_scale
        inputs = torch.cat([x_norm, t_norm], dim=1)
        features = self.feature_extractor(inputs)

        T = self.T_output(features)
        T = T_mean + T_scale * torch.tanh(T / T_scale)
        T = T.clamp(200, 400)

        alpha = self.alpha_output(features)
        alpha = alpha.clamp(1e-6, 1 - 1e-6)

        return T, alpha


model = PropellantPINN(hidden_dim=100, num_layers=4).to(device)


# -------------------------- 4. 损失函数（无修改） --------------------------
def compute_loss(model):
    T_pred_init, alpha_pred_init = model(x_init, t_init)
    loss_init_T = nn.MSELoss()(T_pred_init, T_init)
    loss_init_alpha = nn.MSELoss()(alpha_pred_init, alpha_init)

    T_pred_left, _ = model(x_left, t_left)
    loss_bound_left = nn.MSELoss()(T_pred_left, T_left)

    T_pred_exp, alpha_pred_exp = model(x_exp, t_exp)
    loss_exp_T = nn.MSELoss()(T_pred_exp, T_exp)
    loss_exp_alpha = nn.MSELoss()(alpha_pred_exp, alpha_exp)

    loss_data = (loss_init_T + loss_init_alpha + loss_bound_left + loss_exp_T + loss_exp_alpha) * weight_data

    x_col_ = x_col.detach().clone().requires_grad_(True)
    t_col_ = t_col.detach().clone().requires_grad_(True)
    T_pred, alpha_pred = model(x_col_, t_col_)

    d_alpha_dt = torch.autograd.grad(
        outputs=alpha_pred, inputs=t_col_,
        grad_outputs=torch.ones_like(alpha_pred, device=device),
        create_graph=True, retain_graph=True, allow_unused=True
    )[0]
    d_alpha_dt = d_alpha_dt if d_alpha_dt is not None else torch.zeros_like(alpha_pred)
    d_alpha_dt = d_alpha_dt.clamp(-1e-3, 1e-3)

    T_clamp = T_pred.clamp(min=200)
    exp_term = torch.exp(-Ea / (R * T_clamp))
    exp_term = exp_term.clamp(0, 1e6)

    alpha_clamp = alpha_pred.clamp(1e-6, 1 - 1e-6)
    kinet_rate = A * exp_term * (alpha_clamp ** m) * ((1 - alpha_clamp) ** n)
    kinet_rate = kinet_rate.clamp(0, 1e3)

    loss_kinetic = nn.MSELoss()(d_alpha_dt, kinet_rate)

    dT_dx = torch.autograd.grad(
        outputs=T_pred, inputs=x_col_,
        grad_outputs=torch.ones_like(T_pred, device=device),
        create_graph=True, retain_graph=True, allow_unused=True
    )[0]
    dT_dx = dT_dx if dT_dx is not None else torch.zeros_like(T_pred)
    dT_dx = dT_dx.clamp(-1e2, 1e2)

    dT_dt = torch.autograd.grad(
        outputs=T_pred, inputs=t_col_,
        grad_outputs=torch.ones_like(T_pred, device=device),
        create_graph=True, retain_graph=True, allow_unused=True
    )[0]
    dT_dt = dT_dt if dT_dt is not None else torch.zeros_like(T_pred)
    dT_dt = dT_dt.clamp(-1e-1, 1e-1)

    d2T_dx2 = torch.autograd.grad(
        outputs=dT_dx, inputs=x_col_,
        grad_outputs=torch.ones_like(dT_dx, device=device),
        create_graph=True, allow_unused=True
    )[0]
    d2T_dx2 = d2T_dx2 if d2T_dx2 is not None else torch.zeros_like(dT_dx)
    d2T_dx2 = d2T_dx2.clamp(-1e4, 1e4)

    k = k0 + (k1 - k0) * alpha_pred
    k = k.clamp(0.1, 1.0)
    q_react = rho * Q * d_alpha_dt
    q_react = q_react.clamp(-1e6, 1e6)

    k_dT_dx = k * dT_dx
    d_k_dT_dx_dx = torch.autograd.grad(
        outputs=k_dT_dx, inputs=x_col_,
        grad_outputs=torch.ones_like(k_dT_dx, device=device),
        create_graph=True, allow_unused=True
    )[0]
    d_k_dT_dx_dx = d_k_dT_dx_dx if d_k_dT_dx_dx is not None else torch.zeros_like(k_dT_dx)
    d_k_dT_dx_dx = d_k_dT_dx_dx.clamp(-1e4, 1e4)

    pde_residual_T = rho * cp * dT_dt - d_k_dT_dx_dx - q_react
    pde_residual_T = pde_residual_T.clamp(-1e7, 1e7)
    loss_heat = nn.MSELoss()(pde_residual_T, torch.zeros_like(pde_residual_T))

    x_right_ = x_right.detach().clone().requires_grad_(True)
    t_right_ = t_right.detach().clone().requires_grad_(True)
    T_pred_right, alpha_pred_right = model(x_right_, t_right_)
    dT_dx_right = torch.autograd.grad(
        outputs=T_pred_right, inputs=x_right_,
        grad_outputs=torch.ones_like(T_pred_right, device=device),
        create_graph=True, allow_unused=True
    )[0]
    dT_dx_right = dT_dx_right if dT_dx_right is not None else torch.zeros_like(T_pred_right)

    k_right = k0 + (k1 - k0) * alpha_pred_right
    k_right = k_right.clamp(0.1, 1.0)
    conv_residual = k_right * dT_dx_right - h * (T_pred_right - T_env)
    conv_residual = conv_residual.clamp(-1e3, 1e3)
    loss_conv = nn.MSELoss()(conv_residual, torch.zeros_like(conv_residual))

    loss_pde = (loss_heat + loss_kinetic + loss_conv) * weight_pde
    total_loss = loss_data + loss_pde

    if torch.isnan(total_loss):
        total_loss = torch.tensor(1e6, device=device)
    if torch.isnan(loss_data):
        loss_data = torch.tensor(1e6, device=device)
    if torch.isnan(loss_pde):
        loss_pde = torch.tensor(1e6, device=device)

    return total_loss, loss_data, loss_pde, loss_heat, loss_kinetic


# -------------------------- 5. 训练过程（无修改） --------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

loss_history = []
loss_data_history = []
loss_pde_history = []

print("开始训练PINN模型...")
for epoch in range(epochs):
    optimizer.zero_grad()
    total_loss, loss_data, loss_pde, loss_heat, loss_kinetic = compute_loss(model)

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

    total_loss.backward()
    optimizer.step()
    scheduler.step()

    total_loss_val = total_loss.item() if not torch.isnan(total_loss) else 1e6
    loss_data_val = loss_data.item() if not torch.isnan(loss_data) else 1e6
    loss_pde_val = loss_pde.item() if not torch.isnan(loss_pde) else 1e6

    loss_history.append(total_loss_val)
    loss_data_history.append(loss_data_val)
    loss_pde_history.append(loss_pde_val)

    if (epoch + 1) % 500 == 0:
        loss_heat_val = loss_heat.item() if not torch.isnan(loss_heat) else 1e6
        loss_kinetic_val = loss_kinetic.item() if not torch.isnan(loss_kinetic) else 1e6
        print(f"Epoch [{epoch + 1}/{epochs}], Total Loss: {total_loss_val:.6f}, "
              f"Data Loss: {loss_data_val:.6f}, PDE Loss: {loss_pde_val:.6f}, "
              f"Heat Loss: {loss_heat_val:.6f}, Kinetic Loss: {loss_kinetic_val:.6f}")


# -------------------------- 6. 结果可视化（核心：修复图例） --------------------------
def plot_results():
    x_plot = torch.linspace(0, L, 100).unsqueeze(1).to(device)
    t_plot = torch.linspace(0, 3600, 100).unsqueeze(1).to(device)
    X, T = torch.meshgrid(x_plot.squeeze(), t_plot.squeeze(), indexing="ij")
    X_flat = X.flatten().unsqueeze(1)
    T_flat = T.flatten().unsqueeze(1)

    model.eval()
    with torch.no_grad():
        T_pred_flat, alpha_pred_flat = model(X_flat, T_flat)
        T_pred = T_pred_flat.reshape(X.shape) - 273.15
        alpha_pred = alpha_pred_flat.reshape(X.shape)

    X_np = X.detach().cpu().numpy() * 100
    T_np = T.detach().cpu().numpy() / 3600
    T_pred_np = T_pred.detach().cpu().numpy()
    alpha_pred_np = alpha_pred.detach().cpu().numpy()

    cmap_alpha = LinearSegmentedColormap.from_list("alpha", ["#0000FF", "#FF0000"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ========== 修复1：Loss曲线确保label有效，且legend正常显示 ==========
    ax0 = axes[0, 0]
    ax0.plot(loss_history, label="Total Loss", linewidth=2)
    ax0.plot(loss_data_history, label="Data Loss", linewidth=2, linestyle="--")
    ax0.plot(loss_pde_history, label="PDE Loss", linewidth=2, linestyle="-.")
    ax0.set_yscale("log")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.legend(loc='upper right', frameon=True)  # 显式指定位置，确保图例生效
    ax0.set_title("Training Loss History")

    # 温度场热力图（无图例问题）
    ax1 = axes[0, 1]
    contour_T = ax1.contourf(T_np, X_np, T_pred_np, cmap="jet", levels=30)
    plt.colorbar(contour_T, ax=ax1, label="Temperature (℃)")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Position (cm)")
    ax1.set_title("Temperature Distribution (PINN Prediction)")

    # 固化度场热力图（无图例问题）
    ax2 = axes[1, 0]
    contour_alpha = ax2.contourf(T_np, X_np, alpha_pred_np, cmap=cmap_alpha, levels=30)
    plt.colorbar(contour_alpha, ax=ax2, label="Cure Degree")
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("Position (cm)")
    ax2.set_title("Cure Degree Distribution (PINN Prediction)")

    # ========== 修复2：双轴绘图的图例合并显示 ==========
    ax3 = axes[1, 1]
    t_target = 3600.0
    t_target_tensor = torch.full_like(x_plot, t_target).to(device)
    T_target, alpha_target = model(x_plot, t_target_tensor)

    x_plot_np = x_plot.detach().cpu().numpy() * 100
    T_target_np = T_target.detach().cpu().numpy() - 273.15
    alpha_target_np = alpha_target.detach().cpu().numpy()

    # 主轴线（温度）：显式设置label
    line1 = ax3.plot(x_plot_np, T_target_np, color='blue', linewidth=2,
                     label=f"Temperature (t={t_target / 3600:.1f}h)")[0]
    ax3.set_xlabel("Position (cm)")
    ax3.set_ylabel("Temperature (℃)", color="blue")
    ax3.tick_params(axis='y', labelcolor='blue')

    # 副轴线（固化度）：显式设置label
    ax3_twin = ax3.twinx()
    line2 = ax3_twin.plot(x_plot_np, alpha_target_np, color='red', linewidth=2,
                          label=f"Cure Degree (t={t_target / 3600:.1f}h)")[0]
    ax3_twin.set_ylabel("Cure Degree", color="red")
    ax3_twin.tick_params(axis='y', labelcolor='red')

    # 合并主副轴的图例元素，统一显示
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left', frameon=True)  # 显式传入线条和标签
    ax3.set_title(f"Temperature & Cure Degree at t={t_target / 3600:.1f}h")

    plt.tight_layout()
    plt.show()


plot_results()


# -------------------------- 7. 敏感性分析（无修改） --------------------------
def sensitivity_analysis():
    params = {
        "T_heat": [333.15, 343.15, 353.15, 363.15, 373.15],
        "Ea": [60000.0, 70000.0, 80000.0, 90000.0, 100000.0],
        "k0": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    t_final = 3600.0
    x_mid = torch.tensor([[L / 2]]).to(device)
    t_final_tensor = torch.full_like(x_mid, t_final).to(device)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (param_name, param_values) in enumerate(params.items()):
        alpha_results = []
        T_results = []

        for param_val in param_values:
            globals()[param_name] = param_val

            model.eval()
            with torch.no_grad():
                T_pred, alpha_pred = model(x_mid, t_final_tensor)
                T_results.append(T_pred.detach().cpu().item() - 273.15)
                alpha_results.append(alpha_pred.detach().cpu().item())

        ax = axes[idx]
        # 显式设置label，避免图例为空
        line1 = ax.plot(param_values, alpha_results, marker="o", linewidth=2, color="red",
                        label="Final Cure Degree")[0]
        ax.set_xlabel(param_name)
        ax.set_ylabel("Final Cure Degree (x=0.01m, t=1h)")
        ax.set_title(f"Effect of {param_name} on Cure Degree")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True)  # 显式调用legend

        ax2 = ax.twinx()
        line2 = ax2.plot(param_values, T_results, marker="s", linewidth=2, color="blue", linestyle="--",
                         label="Final Temperature")[0]
        ax2.set_ylabel("Final Temperature (℃)", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")
        ax2.legend(loc='upper right', frameon=True)  # 副轴单独显示图例

    plt.tight_layout()
    plt.show()


sensitivity_analysis()