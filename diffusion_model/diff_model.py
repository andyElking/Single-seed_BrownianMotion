import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import diffrax
from typing import Tuple, Optional, Callable
from functools import partial
import numpy as np


# ============== Diffusion SDE Configuration ==============

class VPSDE:
    """Variance Preserving SDE for diffusion models."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        """Linear noise schedule."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def marginal_stats(self, t):
        """Compute alpha(t) and sigma(t) for the marginal distribution."""
        log_alpha = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        alpha = jnp.exp(log_alpha)
        sigma = jnp.sqrt(1 - alpha ** 2)
        return alpha, sigma

    def drift(self, x, t):
        """Drift coefficient for forward SDE."""
        return -0.5 * self.beta(t) * x

    def diffusion(self, t):
        """Diffusion coefficient for forward SDE."""
        return jnp.sqrt(self.beta(t))

    def reverse_drift(self, x, t, score_fn):
        """Drift for reverse-time SDE."""
        drift = -0.5 * self.beta(t) * x
        diffusion_sq = self.beta(t)
        return drift - diffusion_sq * score_fn(x, t)


# ============== Neural Network Architecture ==============

class TimeEmbedding(eqx.Module):
    """Sinusoidal time embeddings."""
    mlp: eqx.nn.Sequential
    embeddings: jnp.ndarray

    def __init__(self, dim: int, *, key: jax.random.PRNGKey):
        half_dim = dim // 2
        embeddings = jnp.exp(jnp.arange(half_dim) * -(np.log(10000) / (half_dim - 1)))
        self.embeddings = embeddings

        keys = jax.random.split(key, 3)
        self.mlp = eqx.nn.Sequential([
            eqx.nn.Linear(dim, dim * 4, key=keys[0]),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(dim * 4, dim * 4, key=keys[1]),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(dim * 4, dim, key=keys[2])
        ])

    def __call__(self, t):
        emb = t * self.embeddings
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return self.mlp(emb)


class ResNetBlock(eqx.Module):
    """ResNet block with GroupNorm and time conditioning."""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    time_emb_proj: eqx.nn.Linear
    shortcut: Optional[eqx.nn.Conv2d]
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 4)

        self.conv1 = eqx.nn.Conv2d(in_channels, out_channels, 3, padding=1, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(out_channels, out_channels, 3, padding=1, key=keys[1])
        self.time_emb_proj = eqx.nn.Linear(time_emb_dim, out_channels, key=keys[2])

        if in_channels != out_channels:
            self.shortcut = eqx.nn.Conv2d(in_channels, out_channels, 1, key=keys[3])
        else:
            self.shortcut = None

        self.norm1 = eqx.nn.GroupNorm(32, in_channels)
        self.norm2 = eqx.nn.GroupNorm(32, out_channels)

    def __call__(self, x, time_emb):
        h = x
        print(f"Input shape: {h.shape}, Group channels: {self.norm1.channels}")
        h = self.norm1(h)
        h = jax.nn.silu(h)
        h = self.conv1(h)

        # Add time embedding
        time_proj = self.time_emb_proj(time_emb)
        h = h + time_proj.reshape(-1, 1, 1)

        h = self.norm2(h)
        h = jax.nn.silu(h)
        h = self.conv2(h)

        if self.shortcut is not None:
            x = self.shortcut(x)

        return h + x


class AttentionBlock(eqx.Module):
    """Multi-head self-attention block."""
    norm: eqx.nn.GroupNorm
    qkv_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    num_heads: int
    scale: float

    def __init__(self, channels: int, num_heads: int = 8, *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 2)
        self.norm = eqx.nn.GroupNorm(32, channels)
        self.qkv_proj = eqx.nn.Linear(channels, channels * 3, key=keys[0])
        self.out_proj = eqx.nn.Linear(channels, channels, key=keys[1])
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

    def __call__(self, x):
        c, h, w = x.shape  # channels, height, width
        x_norm = self.norm(x)

        # Reshape to sequence
        x_flat = x_norm.reshape(c, h * w).T  # (h*w, c)

        # Compute QKV
        qkv = jax.vmap(self.qkv_proj)(x_flat)  # (h*w, 3*c)
        qkv = qkv.reshape(h * w, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Attention
        attn = jnp.einsum('thc,shc->hts', q, k) * self.scale
        attn = jax.nn.softmax(attn, axis=-1)

        # Apply attention
        out = jnp.einsum('hts,shc->thc', attn, v)
        out = out.reshape(h * w, c)
        out = jax.vmap(self.out_proj)(out)  # (h*w, c)
        out = out.T.reshape(c, h, w)

        return x + out


class UNetModel(eqx.Module):
    """U-Net architecture for diffusion models."""
    time_embed: TimeEmbedding
    num_res_blocks: int
    channel_mult: Tuple[int, ...]

    # Encoder
    conv_in: eqx.nn.Conv2d
    down_blocks: list

    # Middle
    mid_block1: ResNetBlock
    mid_attn: AttentionBlock
    mid_block2: ResNetBlock

    # Decoder
    up_blocks: list
    conv_out: eqx.nn.Sequential

    def __init__(self, in_channels: int = 3, model_channels: int = 128,
                 out_channels: int = 3, num_res_blocks: int = 2,
                 attention_resolutions: Tuple[int, ...] = (16, 8),
                 channel_mult: Tuple[int, ...] = (1, 2, 3, 4),
                 *, key: jax.random.PRNGKey):

        keys = jax.random.split(key, 20)
        time_emb_dim = model_channels * 4

        # Time embedding
        self.time_embed = TimeEmbedding(time_emb_dim, key=keys[0])

        # Initial convolution
        self.conv_in = eqx.nn.Conv2d(in_channels, model_channels, 3, padding=1, key=keys[1])

        # Encoder
        self.down_blocks = []
        ch = model_channels
        resolution = 128
        key_idx = 2

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            for i in range(num_res_blocks):
                self.down_blocks.append(
                    ResNetBlock(ch, out_ch, time_emb_dim, key=keys[key_idx])
                )
                ch = out_ch
                key_idx += 1

                # Check resolution AFTER this res block, BEFORE potential downsampling
                if resolution in attention_resolutions:
                    self.down_blocks.append(
                        AttentionBlock(ch, key=keys[key_idx])
                    )
                    key_idx += 1

            if level != len(channel_mult) - 1:
                self.down_blocks.append(
                    eqx.nn.Conv2d(ch, ch, 3, stride=2, padding=1, key=keys[key_idx])
                )
                resolution //= 2
                key_idx += 1

        # Middle blocks
        self.mid_block1 = ResNetBlock(ch, ch, time_emb_dim, key=keys[key_idx])
        key_idx += 1
        self.mid_attn = AttentionBlock(ch, key=keys[key_idx])
        key_idx += 1
        self.mid_block2 = ResNetBlock(ch, ch, time_emb_dim, key=keys[key_idx])
        key_idx += 1

        # Decoder
        self.up_blocks = []
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult

            for i in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResNetBlock(ch + out_ch, out_ch, time_emb_dim, key=keys[key_idx])
                )
                ch = out_ch
                key_idx += 1

                if resolution in attention_resolutions:
                    self.up_blocks.append(
                        AttentionBlock(ch, key=keys[key_idx])
                    )
                    key_idx += 1

            if level != 0:
                # Upsample
                self.up_blocks.append(
                    eqx.nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1, key=keys[key_idx])
                )
                resolution *= 2
                key_idx += 1

        # Output projection
        self.conv_out = eqx.nn.Sequential([
            eqx.nn.GroupNorm(32, ch),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Conv2d(ch, out_channels, 3, padding=1, key=keys[-1])
        ])

        # Store architecture info
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks

    def __call__(self, x, t):
        # Time embedding
        time_emb = self.time_embed(t)

        # Initial conv
        h = self.conv_in(x)

        # Encoder with skip connections
        hs = [h]
        block_idx = 0

        for level, mult in enumerate(self.channel_mult):
            for i in range(self.num_res_blocks):
                h = self.down_blocks[block_idx](h, time_emb)
                block_idx += 1

                # Check if attention block exists
                if block_idx < len(self.down_blocks) and isinstance(self.down_blocks[block_idx], AttentionBlock):
                    h = self.down_blocks[block_idx](h)
                    block_idx += 1

                hs.append(h)

            if level != len(self.channel_mult) - 1:
                h = self.down_blocks[block_idx](h)  # Downsample
                block_idx += 1
                hs.append(h)

        # Middle
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        # Decoder
        block_idx = 0
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            for i in range(self.num_res_blocks + 1):
                skip = hs.pop()
                h = jnp.concatenate([h, skip], axis=0)  # This concatenates along channel dimension
                h = self.up_blocks[block_idx](h, time_emb)
                block_idx += 1

                # Check if attention block exists
                if block_idx < len(self.up_blocks) and isinstance(self.up_blocks[block_idx], AttentionBlock):
                    h = self.up_blocks[block_idx](h)
                    block_idx += 1

            if level != 0 and block_idx < len(self.up_blocks):
                h = self.up_blocks[block_idx](h)  # Upsample
                block_idx += 1

        # Output
        h = self.conv_out(h)
        return h


# ============== Training Code ==============

def get_score_fn(model, sde, params):
    """Convert noise prediction model to score function."""

    def score_fn(x, t):
        # Predict noise
        noise_pred = model(x, t)
        # Convert to score
        _, sigma = sde.marginal_stats(t)
        return -noise_pred / sigma

    return score_fn


@eqx.filter_jit
def loss_fn(model, sde, params, batch, key):
    """Compute the diffusion training loss."""
    x0 = batch
    batch_size = x0.shape[0]

    # Sample random timesteps
    key, subkey = jax.random.split(key)
    t = jax.random.uniform(subkey, (batch_size,))

    # Sample noise
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, x0.shape)

    # Get marginal distribution parameters
    alpha, sigma = jax.vmap(sde.marginal_stats)(t)
    alpha = alpha[:, None, None, None]
    sigma = sigma[:, None, None, None]

    # Create noisy samples
    xt = alpha * x0 + sigma * noise

    # Predict noise - vmap over batch dimension
    model_batched = jax.vmap(model, in_axes=(0, 0))
    noise_pred = model_batched(xt, t)

    # MSE loss
    loss = jnp.mean((noise - noise_pred) ** 2)

    return loss


@eqx.filter_jit
def train_step(model, sde, optimizer, params, opt_state, batch, key):
    """Single training step."""
    loss, grads = jax.value_and_grad(loss_fn, argnums=2)(model, sde, params, batch, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# ============== Sampling Code ==============

class ReverseSDE(eqx.Module):
    """Reverse SDE for sampling."""
    sde: VPSDE
    score_fn: Callable

    def __init__(self, sde, score_fn):
        self.sde = sde
        self.score_fn = score_fn

    def vector_field(self, t, y, args):
        """Drift term for reverse SDE (for ODE solving)."""
        x = y
        score = self.score_fn(x, 1.0 - t)  # Reverse time
        drift = self.sde.reverse_drift(x, 1.0 - t, lambda x, t: score)
        return -drift  # Negative because we're going backwards in time

    def diffusion(self, t, y, args):
        """Diffusion term for reverse SDE."""
        return self.sde.diffusion(1.0 - t)


@eqx.filter_jit
def sample_sde(model, sde, reverse_sde, shape, num_steps, key):
    """Sample from the model using SDE solver."""
    # Start from noise
    key, subkey = jax.random.split(key)
    x_init = jax.random.normal(subkey, shape)

    # Create Brownian motion
    key, subkey = jax.random.split(key)
    brownian_motion = diffrax.VirtualBrownianTree(
        t0=0.0, t1=1.0, tol=1e-3, shape=shape, key=subkey
    )

    # Solve reverse SDE
    terms = diffrax.MultiTerm(
        diffrax.ODETerm(reverse_sde.vector_field),
        diffrax.ControlTerm(reverse_sde.diffusion, brownian_motion)
    )

    solver = diffrax.EulerMaruyama()
    saveat = diffrax.SaveAt(t1=True)

    sol = diffrax.diffeqsolve(
        terms, solver,
        t0=0.0, t1=1.0,
        dt0=1.0 / num_steps,
        y0=x_init,
        saveat=saveat,
        max_steps=num_steps
    )

    return sol.ys[-1]


# ============== Main Training Loop ==============

def create_train_state(key, learning_rate=2e-4):
    """Initialize model and optimizer."""
    model = UNetModel(key=key)

    # Create optimizer with cosine schedule
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=100000,
        alpha=0.1
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, b1=0.9, b2=0.999, weight_decay=0.01)
    )

    # Get initial parameters
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(params)

    return model, static, params, optimizer, opt_state


def train(dataset_iterator, num_steps=100000, batch_size=16, key=jax.random.PRNGKey(0)):
    """Main training loop."""
    # Initialize
    key, subkey = jax.random.split(key)
    model, static, params, optimizer, opt_state = create_train_state(subkey)
    sde = VPSDE()

    # Training loop
    for step in range(num_steps):
        batch = next(dataset_iterator)  # Assume normalized to [-1, 1]

        key, subkey = jax.random.split(key)
        params, opt_state, loss = train_step(
            eqx.combine(params, static), sde, optimizer,
            params, opt_state, batch, subkey
        )

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

        # Periodically sample
        if step % 1000 == 0 and step > 0:
            key, subkey = jax.random.split(key)
            model_full = eqx.combine(params, static)
            score_fn = get_score_fn(model_full, sde, params)
            reverse_sde = ReverseSDE(sde, score_fn)

            # Sample a batch
            samples = jax.vmap(
                lambda k: sample_sde(model_full, sde, reverse_sde, (3, 128, 128), 1000, k)
            )(jax.random.split(subkey, 4))

            # samples are now ready for visualization
            print(f"Generated samples at step {step}")

    return eqx.combine(params, static)


# ============== Data Loading Placeholder ==============

def create_dummy_dataset_iterator(batch_size=16):
    """Create a dummy iterator for testing."""
    key = jax.random.PRNGKey(0)
    while True:
        key, subkey = jax.random.split(key)
        # Random images normalized to [-1, 1]
        batch = jax.random.uniform(subkey, (batch_size, 3, 128, 128), minval=-1, maxval=1)
        yield batch


# Example usage:
if __name__ == "__main__":
    # For actual training, replace with real ImageNet loader
    dataset_iter = create_dummy_dataset_iterator(batch_size=16)

    # Train model
    trained_model = train(dataset_iter, num_steps=1000, key=jax.random.PRNGKey(42))