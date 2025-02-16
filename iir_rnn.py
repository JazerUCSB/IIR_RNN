import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.signal as signal
import librosa
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 🚀 Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable CuDNN optimizations for speedup
torch.backends.cudnn.benchmark = True

# 🚀 Load the impulse response from the provided WAV file
filename = "Mesa_OS_4x12_57_m160.wav"
impulse_response, Fs = librosa.load(filename, sr=None)  # Keep original sampling rate

L = len(impulse_response)  # Use full length (~9000 samples)
impulse_response /= np.max(np.abs(impulse_response))  # Normalize

# 🎛 Increase filter order for better approximation
P_target, Q_target = 80, 80  # Higher-order IIR filter

# 🛠 Training Hyperparameters
batch_size = 256
seq_len = len(impulse_response)  # Length of input signals
learning_rate = 0.003
num_epochs = 8192
regularization_weight = 0.005

# 🎼 Generate dry input signals (random noise)
x_train = np.random.randn(batch_size, seq_len)
y_train = np.array([signal.convolve(x, impulse_response, mode='same')[:seq_len] for x in x_train])

# 🔄 Convert to PyTorch tensors and move to GPU
x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)


# 🎯 Levinson-Durbin Recursion for All-Pole Model
def levinson_durbin(r, order):
    """
    Implements the recursive Levinson-Durbin algorithm for solving the Yule-Walker equations
    to compute autoregressive (AR) coefficients.
    """
    a = np.zeros(order + 1)  # AR coefficients
    a[0] = 1.0  # AR(0) is always 1
    e = r[0]  # Initial error (total variance)

    if e == 0:
        raise ValueError("Degenerate autocorrelation: First coefficient is zero.")

    for k in range(1, order + 1):
        lambda_k = -np.dot(a[:k], r[k:0:-1]) / e
        a[1:k+1] += lambda_k * a[k-1::-1]
        e *= (1 - lambda_k**2)

    return a


def compute_all_pole_model(impulse_response, order):
    """Compute all-pole (AR) coefficients using the Levinson-Durbin recursion."""
    r = np.correlate(impulse_response, impulse_response, mode="full")  # Compute autocorrelation
    r = r[len(r)//2:]  # Keep only non-negative lags

    # Compute AR coefficients using Levinson-Durbin
    a_coeffs = levinson_durbin(r, order)
    
    return a_coeffs


# 🎯 Define an Optimized IIR-Structured RNN
class IIRFilterRNN(nn.Module):
    def __init__(self, P, Q, seq_len, a_coeffs=None):
        super(IIRFilterRNN, self).__init__()
        self.P = P  # Feedback (poles)
        self.Q = Q  # Feedforward (zeros)
        self.seq_len = seq_len  # Ensure correct output shape

        # Initialize denominator coefficients (poles)
        if a_coeffs is None:
            self.a = nn.Parameter(torch.randn(1, 1, P, device=device) * 0.1)  # Default random init
        else:
            a_tensor = torch.tensor(-a_coeffs[1:], dtype=torch.float32, device=device)  # Negate for standard form
            self.a = nn.Parameter(a_tensor.view(1, 1, -1))  # Convert to correct shape

        # Initialize zeros randomly
        self.b = nn.Parameter(torch.randn(1, 1, Q + 1, device=device) * 0.1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension (batch, 1, seq_len)

        # Efficient feedforward convolution (like FIR filter)
        ff_out = F.conv1d(x, self.b, padding=self.Q)[:, :, :self.seq_len]

        # Efficient feedback convolution (IIR filter)
        fb_out = F.conv1d(ff_out, self.a, padding=self.P-1)[:, :, :self.seq_len]

        return fb_out.squeeze(1)


# 📌 Compute All-Pole Approximation
a_coeffs = compute_all_pole_model(impulse_response, P_target)

# 📌 Instantiate Model with Precomputed Denominator Coefficients
model = IIRFilterRNN(P=P_target, Q=Q_target, seq_len=seq_len, a_coeffs=a_coeffs).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()


# 🔬 Extract Poles and Zeros
def get_poles_zeros(model):
    """Extract poles and zeros from trained IIR filter."""
    a_coeffs = [1.0] + (-model.a.detach().cpu().numpy().flatten()).tolist()
    b_coeffs = model.b.detach().cpu().numpy().flatten().tolist()

    # Compute poles and zeros
    zeros, poles, _ = signal.tf2zpk(b_coeffs, a_coeffs)
    return zeros, poles


# 🚀 Training Loop (Using Mixed Precision for Faster Training)
scaler = torch.amp.GradScaler()

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    with torch.amp.autocast(device_type="cuda"):
        y_pred = model(x_train_tensor)

        # Compute losses
        loss_mse = loss_fn(y_pred, y_train_tensor)
        loss_freq = torch.mean((torch.abs(torch.fft.rfft(y_pred.float(), dim=-1))-torch.abs(torch.fft.rfft(y_train_tensor.float(), dim=-1)))**2)
        loss_l1 = regularization_weight * (torch.sum(torch.abs(model.a)) + torch.sum(torch.abs(model.b)))

        # Compute pole stability loss
        poles = get_poles_zeros(model)[1]
        loss_stability = torch.sum(torch.clamp(torch.abs(torch.tensor(poles, device=device)) - 1, min=0))

        loss = (0.2 * loss_mse) + (3.0 * loss_freq) + loss_l1 + loss_stability

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if epoch % 50 == 0:
        print(f"✅ Epoch {epoch} | MSE: {loss_mse.item()} | Freq Loss: {loss_freq.item()} | Total Loss: {loss.item()}")


def plot_frequency_response(model, impulse_response, Fs, filename="freq_response.png"):
    """Compute and save the frequency response plot of the trained IIR filter."""
    
    # Extract coefficients and ensure they are 1D NumPy arrays
    a_coeffs = np.array([1.0] + (-model.a.detach().cpu().numpy().flatten()).tolist(), dtype=np.float64)  # ✅ Ensures proper shape
    b_coeffs = np.array(model.b.detach().cpu().numpy().flatten(), dtype=np.float64)  # ✅ Ensures proper shape

    # Debugging print
    print(f"📏 Debugging get_poles_zeros() -> a_coeffs shape: {a_coeffs.shape}, b_coeffs shape: {b_coeffs.shape}")
    print(f"📏  {a_coeffs}, {b_coeffs}")
   # Compute frequency response of trained IIR filter
    w, h_iir = signal.freqz(b_coeffs, a_coeffs, worN=2048, fs=Fs)

    # Compute frequency response of original impulse response
    w_rir, h_rir = signal.freqz(impulse_response, [1.0], worN=2048, fs=Fs)

    # Plot the responses
    plt.figure(figsize=(10, 5))
    plt.plot(w, 20 * np.log10(abs(h_iir)), label="Trained IIR Filter", linestyle='dashed', color='red')
    plt.plot(w_rir, 20 * np.log10(abs(h_rir)), label="Original Impulse Response", linestyle='solid', color='blue')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Frequency Response Comparison')
    plt.legend()
    plt.grid()

    # Save instead of showing the plot
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved frequency response plot as {filename}")

plot_frequency_response(model, impulse_response, Fs)
