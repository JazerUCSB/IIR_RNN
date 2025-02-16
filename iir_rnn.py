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
P_target, Q_target = 100, 100  # Higher-order IIR filter

# 🛠 Training Hyperparameters
batch_size = 200
seq_len = 500  # Length of input signals
learning_rate = 0.001
num_epochs = 5000
regularization_weight = 0.01

# 🎼 Generate dry input signals (random noise)
x_train = np.random.randn(batch_size, seq_len)

# 🎧 Generate convolved output signals (fixed shape)
y_train = np.array([signal.convolve(x, impulse_response, mode='same')[:seq_len] for x in x_train])  # ✅ Fixed

# 🔄 Convert to PyTorch tensors and move to GPU
x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)


# 🎯 Define an Optimized IIR-Structured RNN
class IIRFilterRNN(nn.Module):
    def __init__(self, P, Q, seq_len):
        super(IIRFilterRNN, self).__init__()
        self.P = P  # Feedback (poles)
        self.Q = Q  # Feedforward (zeros)
        self.seq_len = seq_len  # Ensure correct output shape

        # Trainable weights for IIR filter coefficients
        self.a = nn.Parameter(torch.randn(1, 1, P, device=device) * 0.1)  # Poles (feedback)
        self.b = nn.Parameter(torch.randn(1, 1, Q+1, device=device) * 0.1)  # Zeros (feedforward)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension (batch, 1, seq_len)

        # Efficient feedforward convolution (like FIR filter)
        ff_out = F.conv1d(x, self.b, padding=self.Q)[:, :, :self.seq_len]  # Trim to correct output length

        # Efficient feedback convolution (IIR filter)
        fb_out = F.conv1d(ff_out, self.a, padding=self.P-1)[:, :, :self.seq_len]  # Trim again

        return fb_out.squeeze(1)  # Remove channel dim

# 📌 Instantiate Model and Move to GPU
model = IIRFilterRNN(P=P_target, Q=Q_target, seq_len=seq_len).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

def frequency_loss(y_pred, y_true):
    """ Compute the frequency response error between the model and the target IR. """
    
    # Convert to float32 before applying FFT (Fix for cuFFT)
    y_pred = y_pred.float()
    y_true = y_true.float()

    y_pred_fft = torch.fft.rfft(y_pred, dim=-1)
    y_true_fft = torch.fft.rfft(y_true, dim=-1)

    # Compare Magnitude & Phase
    mag_loss = torch.mean((torch.abs(y_pred_fft) - torch.abs(y_true_fft)) ** 2)
    phase_loss = torch.mean((torch.angle(y_pred_fft) - torch.angle(y_true_fft)) ** 2)

    return mag_loss + phase_loss  # ✅ Runs in full precision


def get_poles_zeros(model):
    """Extract poles and zeros from trained IIR filter."""
    a_coeffs = [1.0] + (-model.a.detach().cpu().numpy().flatten()).tolist()  # ✅ Convert to 1D list
    b_coeffs = model.b.detach().cpu().numpy().flatten().tolist()  # ✅ Convert to 1D list

    # Ensure coefficients are NumPy arrays
    a_coeffs = np.array(a_coeffs, dtype=np.float64)  # ✅ Ensure correct dtype
    b_coeffs = np.array(b_coeffs, dtype=np.float64)  # ✅ Ensure correct dtype

    # Compute poles and zeros
    zeros, poles, _ = signal.tf2zpk(b_coeffs, a_coeffs)
    return zeros, poles


# 🔬 Ensure stability by penalizing poles outside the unit circle
def pole_stability_loss(model):
    poles = get_poles_zeros(model)[1]  # Extract poles
    return torch.sum(torch.clamp(torch.abs(torch.tensor(poles, device=device)) - 1, min=0))  # Penalize unstable poles

# 🚀 Training Loop (Using Mixed Precision for Faster Training)
# 🚀 Use automatic mixed precision for speedup
scaler = torch.amp.GradScaler()  # ✅ Fixed

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    with torch.amp.autocast(device_type="cuda"):  # Mixed precision
        y_pred = model(x_train_tensor)

        # Compute losses
        loss_mse = loss_fn(y_pred, y_train_tensor)  # Time domain loss
        loss_freq = frequency_loss(y_pred, y_train_tensor)  # Frequency domain loss

        # Adjust loss weighting (increase frequency loss importance)
        loss_l1 = regularization_weight * (torch.sum(torch.abs(model.a)) + torch.sum(torch.abs(model.b)))
        loss_stability = 10.0 * pole_stability_loss(model)

        # Weight frequency loss higher than MSE
        loss = (0.2 * loss_mse) + (3.0 * loss_freq) + loss_l1 + loss_stability  # 🔥 Increase freq loss weight

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if epoch % 50 == 0:
        print(f"✅ Epoch {epoch} Completed! MSE: {loss_mse.item()}, Freq: {loss_freq.item()}, Total Loss: {loss.item()}")



def plot_frequency_response(model, impulse_response, Fs, filename="freq_response.png"):
    """Compute and save the frequency response plot of the trained IIR filter."""
    
    # Extract coefficients and ensure they are 1D NumPy arrays
    a_coeffs = np.array([1.0] + (-model.a.detach().cpu().numpy().flatten()).tolist(), dtype=np.float64)  # ✅ Ensures proper shape
    b_coeffs = np.array(model.b.detach().cpu().numpy().flatten(), dtype=np.float64)  # ✅ Ensures proper shape

    # Debugging print
    print(f"📏 Debugging get_poles_zeros() -> a_coeffs shape: {a_coeffs.shape}, b_coeffs shape: {b_coeffs.shape}")

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
