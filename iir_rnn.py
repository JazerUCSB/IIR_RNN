import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.signal as signal
import librosa

# Load the system's impulse response
filename = "wideband_audio.wav"
audio, Fs = librosa.load(filename, sr=None)  

L = 128  # Full impulse response length
impulse_response = audio[:L] / np.max(np.abs(audio[:L]))  # Normalize

# Generate dry signals (random noise)
batch_size = 10
seq_len = 500  # Length of input signals
P_target, Q_target = 6, 6  # Lower-order approximation (desired IIR order)
num_epochs = 500
learning_rate = 0.005
regularization_weight = 0.01  # L1 weight to enforce lower order

x_train = np.random.randn(batch_size, seq_len)
y_train = np.array([signal.convolve(x, impulse_response, mode='same') for x in x_train])

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Define an IIR-Structured RNN for Lower-Order Approximation
class IIRFilterRNN(nn.Module):
    def __init__(self, P, Q):
        super(IIRFilterRNN, self).__init__()
        self.P = P  # Lower-order Feedback (poles)
        self.Q = Q  # Lower-order Feedforward (zeros)

        # Trainable weights representing the IIR filter coefficients
        self.a = nn.Parameter(torch.randn(P) * 0.1)  # Poles
        self.b = nn.Parameter(torch.randn(Q+1) * 0.1)  # Zeros

    def forward(self, x):
        batch_size, seq_len = x.shape
        y = torch.zeros(batch_size, seq_len, device=x.device)

        for n in range(seq_len):
            ff_sum = sum(self.b[j] * x[:, n-j] if n-j >= 0 else 0 for j in range(self.Q+1))
            fb_sum = sum(self.a[i] * y[:, n-i-1] if n-i-1 >= 0 else 0 for i in range(self.P))
            y[:, n] = ff_sum + fb_sum
        
        return y

# Instantiate Model with Lower Order
model = IIRFilterRNN(P_target, Q_target)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Define a function to compute frequency response loss
def frequency_loss(y_pred, y_true):
    """ Compute the frequency response error between the model and the target RIR. """
    y_pred_fft = torch.fft.rfft(y_pred, dim=-1)
    y_true_fft = torch.fft.rfft(y_true, dim=-1)
    
    # Compare Magnitude & Phase
    mag_loss = torch.mean((torch.abs(y_pred_fft) - torch.abs(y_true_fft)) ** 2)
    phase_loss = torch.mean((torch.angle(y_pred_fft) - torch.angle(y_true_fft)) ** 2)
    
    return mag_loss + phase_loss

# Training Loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    y_pred = model(x_train_tensor)
    
    # MSE Loss (Time Domain)
    loss_mse = loss_fn(y_pred, y_train_tensor)
    
    # Frequency Domain Loss (to match impulse response)
    loss_freq = frequency_loss(y_pred, y_train_tensor)
    
    # Regularization Loss to Enforce Lower-Order Approximation
    loss_l1 = regularization_weight * (torch.sum(torch.abs(model.a)) + torch.sum(torch.abs(model.b)))

    # Total Loss
    loss = loss_mse + loss_freq + loss_l1
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}, MSE: {loss_mse.item()}, Freq: {loss_freq.item()}, L1: {loss_l1.item()}")

# Extract Poles and Zeros of the Trained Lower-Order IIR Model
def get_poles_zeros(model):
    a_coeffs = [1.0] + (-model.a.detach().cpu().numpy()).tolist()
    b_coeffs = model.b.detach().cpu().numpy().tolist()
    zeros, poles, _ = signal.tf2zpk(b_coeffs, a_coeffs)
    return zeros, poles

zeros, poles = get_poles_zeros(model)
print("Learned Zeros:", zeros)
print("Learned Poles:", poles)
